/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer tensor_trainer sub-plugin for nntrainer
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file   tensor_trainer_nntrainer.cc
 * @date   02 Dec 2022
 * @brief  NNStreamer tensor_trainer sub-plugin
 * @see    http://github.com/nnstreamer/nntrainer
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Hyunil Park <hyunil46.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

/**
 * # Action description and constraints
 * tensor_trainer_nntrainer.cc is a nnstreamer sub-plugin be used by
 * nnstreamer(tensor_trainer) for training model.
 *
 * ## Notice
 * 1. Models are limited to creating with configuration files.
 * 2. sub-plugin receives only pre-processed tensor data.
 * 3. The current feature behavior has been tested with MNIST.
 * 4. mnist.json has 'gst_caps' containing the information below.
 * "gst_caps":"other/tensors, format=(string)static, framerate=(fraction)0/1,
 *  num_tensors=(int)2, dimensions=(string)1:1:784:1.1:1:10:1,
 *  types=(string)float32.float32"
 *
 * ## Example launch line is as below
 * gst-launch-1.0 datareposrc location=mnist_trainingSet.dat json=mnist.json \
 * start-sample-index=3 stop-sample-index=202 epochs=5 ! \
 * tensor_trainer framework=nntrainer model-config=mnist.ini \
 * model-save-path=model.bin num-inputs=1 num-labels=1 \
 * num-training-samples=100 num-validation-samples=100 epochs=5 ! tensor_sink

 * ## Notice below item in gst_caps
 * 1. input-dims:input dimensions, in case of MNIST, 1:1:784:1 is a input and
 *   1:1:10:1 is a label.
 * 2. input_type: data type. Sample(mnist_trainingSet.dat)'s data type is
 *   float32.
 *
 * ## Notice the tensor_trainer's properties.
 * 1. model-config: model configuration file path. models are limited to
 *   creating with configuration files.
 * 2. model-save-path: model save path by query in MLOps
 * 3. num-inputs: sub-plugin supports multiple inputs, in case of MNIST,
 *   num-inputs is 1.
 * 4. num-labels: sub-plugin supports multiple labels, in case of MNIST,
 *   num-labels is 1.
 * 5. num-training-samples: Number of training samples, A sample can consist of
 *   multiple inputs and labels in tensors(in case of MNIST, all is 1), set how
 *   many samples are taken for training model.
 * 6. num-validation-samples: num-validation-samples, A sample can consist of
 *   multiple inputs and labels in tensors(in case of MNIST, all is 1), set how
 *   many samples are taken for validation model.
 * 7. epochs: epochs are repetitions of training samples and validation
 *   samples. number of samples received for model training is
 *   (num-training-samples + num-validation-samples) * epochs
 *
 * ## Action
 * When a sub-plugin is loaded at runtime,
 * NNTrainerImpl's model_config, num_inputs, num_labels, num_training_samples,
 * num_validation_samples, model_save_path, model_load_path and num_epochs is
 * set by
 * setNNStreamerProperties() and each tensor size is set.
 * For MNIST test, mnist_trainingSet.dat is used,
 * it consist of 1000 samples and it has 1 inputs, 1 labels, 784 and 10 size of
 * float32.
 * now, sub-plugin get num_tensors:2, tensor_size[0]:3136,
 * tensor_size[1]:40, num_inputs:1, num_labels:1, num_training_samples:100,
 * num_validation_samples:100, num_epochs:5 from nnstreamer(tensor_trainer).
 * Total number of sample to be received in sub-plugin is 1000, (100 + 100) * 5.
 *
 * nnstreamer(tensor_trainer) push a sample by call nntrainer_model_push_data()
 * and sub-plugin copy a sample to queue with incrementing
 * TensorsQueue::push_count. 100 samples for training, 100 samples for
 * validation comes in order.
 *
 * trainingDataGenCb() and validationDataGenCb callback is called
 * when nntrainer needs sample for training.
 * sub-plugin copy a sample from queue to '**inputs' and '**labels'
 * with incrementing TensorsQueue::pop_count.
 * if TensorsQueue::pop_count is same with total_num_samples then
 * '*last' is set true, and 1 epochs is finished.
 *
 */

#include "tensor_trainer_nntrainer.hh"
#include <cstring>
#include <inttypes.h>
#include <iostream>
#include <limits.h>
#include <nntrainer_log.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

/**
 * @brief startup constructor
 */
void init_subplugin_nntrainer(void) __attribute__((constructor));

/**
 * @brief startdown destructor
 */
void fini_subplugin_nntrainer(void) __attribute__((destructor));

void nntrainer_thread_func(NNTrainer::NNTrainerImpl *nntrainer) {
  nntrainer->trainModel();
}

/**
 * @brief Check if queue is empty
 */
bool NNTrainer::TensorsQueue::isQueueEmpty() {
  ml_logd("front:%d, rear:%d", queue_front, queue_rear);
  if (queue_front == queue_rear) {
    ml_logd("queue is empty ");
    return TRUE;
  }
  return FALSE;
}

/**
 * @brief Check if queue is full
 */
bool NNTrainer::TensorsQueue::isQueueFull() {
  if (((queue_rear + 1) % queue_size) == queue_front) {
    ml_logd("queue is full");
    return TRUE;
  }
  return FALSE;
}

/**
 * @brief push_data function
 * tensor_trainer call this function to push tensor data.
 * For epoch, (number of train samples + number of valid samples) * epoch
 * data should be received.
 * Sub-plugin don't keep dataset for epoch.
 */
static int nntrainer_model_push_data(const GstTensorTrainerFramework *fw,
                                     const GstTensorTrainerProperties *prop,
                                     void *private_data,
                                     const GstTensorMemory *input) {
  NNTrainer::TensorsQueue *tensors_queue = nullptr;
  NNTrainer::NNTrainerImpl *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerImpl *>(private_data);
  UNUSED(fw);
  UNUSED(prop);
  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);

  ml_logd("<called>");
  ml_logd("pid[%d], tid[%d]", pid, tid);

  if (!nntrainer) {
    ml_loge("Failed get nntrainer");
    return -1;
  }

  nntrainer->incrementNumOfPushedSamples();
  ml_logd("Received data (%d/%d(total))", nntrainer->getNumOfPushedSamples(),
          nntrainer->getNumOfTotalSamples());
  if (nntrainer->getNumOfTotalSamples() < nntrainer->getNumOfPushedSamples()) {
    ml_logd("Already received all data required for the train, "
            "but push_data is called");
    return 0;
  }

  if (nntrainer->train_tensors_queue->getPushCount() <
      nntrainer->getNumOfTrainingSamples()) {
    tensors_queue = nntrainer->train_tensors_queue.get();
    ml_logd("#### T-Data ####");
  } else if (nntrainer->valid_tensors_queue->getPushCount() <
             nntrainer->getNumOfValidationSamples()) {
    tensors_queue = nntrainer->valid_tensors_queue.get();
    ml_logd("#### V-Data ####");
  } else {
    ml_loge("Invalid push_count");
    return -1;
  }

  tensors_queue->push(input);

  ml_logd("T-pushed: %d/%d, V-pushed:%d/%d\n",
          nntrainer->train_tensors_queue->getPushCount(),
          nntrainer->getNumOfTrainingSamples(),
          nntrainer->valid_tensors_queue->getPushCount(),
          nntrainer->getNumOfValidationSamples());
  ml_logd("<leaved>");
  return 0;
}

void NNTrainer::TensorsQueue::push(const GstTensorMemory *input) {
  ml_logd("<called>");

  std::unique_lock<std::mutex> lock(queue_lock);
  data_full.wait(lock, [&] { return !isQueueFull(); });
  ml_logd("nntrainer_model_push_data condition is met");

  unsigned int idx = 0, i = 0;
  i = queue_rear;
  ml_logd("Insert, queue_rear : %d", i);
  for (auto inputs : queue[i].inputs) {
    ml_logd("inputs : %p", inputs);
    ml_logd("input[%d]:%p, size:%zd\n", idx, input[idx].data, input[idx].size);
    std::memcpy(inputs, input[idx].data, input[idx].size);
    idx++;
  }

  for (auto labels : queue[i].labels) {
    ml_logd("labels : %p", labels);
    ml_logd("input[%d]:%p, size:%zd\n", idx, input[idx].data, input[idx].size);
    std::memcpy(labels, input[idx].data, input[idx].size);
    idx++;
  }

  push_count++;
  queue_rear = (queue_rear + 1) % queue_size;
  int queue_count = (queue_rear - queue_front + queue_size) % queue_size;

  ml_logd("front:%d, rear:%d, filled:%d", queue_front, queue_rear, queue_count);
  ml_logd("(pop/push: %d/%d)", pop_count, push_count);
  lock.unlock();
  data_empty.notify_one();

  ml_logd("<leave>");
}

void NNTrainer::TensorsQueue::pop(float **input, float **label, bool *last) {
  ml_logd("<called>");

  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);
  ml_logd("pid[%d], tid[%d]", pid, tid);

  std::unique_lock<std::mutex> lock(queue_lock);
  ml_logd("(pop/push: %d/%d)", pop_count, push_count);
  data_empty.wait(lock, [this] { return !isQueueEmpty(); });
  ml_logd("pop condition is met");

  ml_logd("num_of_inputs: %d, num_of_labels: %d", num_of_inputs, num_of_labels);

  unsigned int i = 0;
  unsigned int idx = queue_front;
  ml_logd("Delete, queue_front: %d", idx);

  for (i = 0; i < num_of_inputs; i++) {
    ml_logd("memcpy Addr %p, %p, size=%d\n", *(input + i), queue[idx].inputs[i],
            input_size[i]);
    std::memcpy(*(input + i), queue[idx].inputs[i], input_size[i]);
  }
  for (i = 0; i < num_of_labels; i++) {
    ml_logd("memcpy Addr %p, %p, size=%d", *(label + i), queue[idx].labels[i],
            label_size[i]);
    std::memcpy(*(label + i), queue[idx].labels[i], label_size[i]);
  }

  pop_count++;
  queue_front = (queue_front + 1) % queue_size;

  ml_logd("(pop/push: %d/%d)", pop_count, push_count);

  if (pop_count < num_of_samples) { // train or valid num samples
    *last = false;
  } else {
    *last = true;
    pop_count = 0;
  }

  int queue_count = (queue_rear - queue_front + queue_size) % queue_size;
  ml_logd("front:%d, rear:%d, filled:%d", queue_front, queue_rear, queue_count);
  lock.unlock();
  data_full.notify_one();

  ml_logd("<leave>");
  return;
}

/**
 * @brief  nntrainer calls when it needs data.
 */
int trainingDataGenCb(float **input, float **label, bool *last,
                      void *user_data) {
  auto tensors_queue = reinterpret_cast<NNTrainer::TensorsQueue *>(user_data);
  if (!tensors_queue) {
    ml_loge("invalid user_data");
    return 0;
  }
  ml_logd("trainingDataGenCb");
  tensors_queue->pop(input, label, last);
  return 0;
}

/**
 * @brief  nntrainer calls when it needs data.
 */
int validationDataGenCb(float **input, float **label, bool *last,
                        void *user_data) {
  auto tensors_queue = reinterpret_cast<NNTrainer::TensorsQueue *>(user_data);
  if (!tensors_queue) {
    ml_loge("invalid user_data");
    return 0;
  }
  ml_logd("validationDataGenCb");
  tensors_queue->pop(input, label, last);

  return 0;
}

void NNTrainer::NNTrainerImpl::createDataset() {

  ml_logd("<called>");

  train_tensors_queue = std::make_unique<NNTrainer::TensorsQueue>(
    num_training_samples, num_inputs, num_labels, tensors_inputsize);
  valid_tensors_queue = std::make_unique<NNTrainer::TensorsQueue>(
    num_validation_samples, num_inputs, num_labels, tensors_inputsize);

  try {
    if (num_training_samples) {
      dataset_train =
        ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                 trainingDataGenCb, train_tensors_queue.get());
      model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    }
    if (num_validation_samples) {
      dataset_valid = ml::train::createDataset(
        ml::train::DatasetType::GENERATOR, validationDataGenCb,
        valid_tensors_queue.get());
      model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_valid);
    }
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  ml_logd("<leave>");
}

void NNTrainer::NNTrainerImpl::getRunStats() {
  ml_logd("<called>");

  train_stats = model->getTrainingStats();
  valid_stats = model->getValidStats();

  ml_logd("<leave>");
}

NNTrainer::TensorsQueue::TensorsQueue(unsigned int _num_of_samples,
                                      unsigned int _num_of_inputs,
                                      unsigned int _num_of_labels,
                                      unsigned int _tensors_size[]) :
  num_of_samples{_num_of_samples},
  num_of_inputs{_num_of_inputs},
  num_of_labels{_num_of_labels} {

  ml_logd("<called>");
  const unsigned int min_queue_size = 30;
  queue_size =
    (_num_of_samples > min_queue_size) ? min_queue_size : _num_of_samples;
  ml_logd("queue_size:%d", queue_size);
  queue.reserve(queue_size);

  unsigned int idx = 0, i = 0;
  for (i = 0; i < num_of_inputs; i++) {
    input_size[i] = _tensors_size[idx++];
    ml_logd("input_size[%d]=%d", i, input_size[i]);
  }
  for (i = 0; i < num_of_labels; i++) {
    label_size[i] = _tensors_size[idx++];
    ml_logd("label_size[%d]=%d", i, label_size[i]);
  }

  unsigned int cur_queue_size = 0;

  /* make queue */
  while (cur_queue_size < queue_size) {
    NNTrainer::TensorsData t_data;
    unsigned int i = 0;
    char *p_data = nullptr;
    for (i = 0; i < num_of_inputs; i++) {
      p_data = new char[input_size[i]];
      t_data.inputs.emplace_back(p_data);
    }
    for (i = 0; i < num_of_labels; i++) {
      p_data = new char[label_size[i]];
      t_data.labels.emplace_back(p_data);
    }

    queue.emplace_back(t_data);
    cur_queue_size++;
  }
  ml_logd("<leave>");
}

NNTrainer::TensorsQueue::~TensorsQueue() {
  g_print("%s:%d:%s: <called>\n", __FILE__, __LINE__, __func__);

  for (auto &data : queue) {
    for (auto inputs : data.inputs) {
      ml_logd("free: ##I addr:%p", inputs);
      delete[] inputs;
    }
    for (auto labels : data.labels) {
      ml_logd("free: ##L addr:%p", labels);
      delete[] labels;
    }
  }
}

void NNTrainer::NNTrainerImpl::setNNStreamerProperties(
  const GstTensorTrainerProperties *prop) {

  unsigned int i;
  ml_logd("<called>");

  num_tensors = prop->input_meta.num_tensors;
  ml_logd("num_tensors: %d", num_tensors);

  for (i = 0; i < num_tensors; i++) {
    tensors_inputsize[i] = gst_tensor_info_get_size(&prop->input_meta.info[i]);
    ml_logd("tensors_inputsize[%d]:%d", i, tensors_inputsize[i]);
  }
  num_inputs = prop->num_inputs;
  num_labels = prop->num_labels;
  num_training_samples = prop->num_training_samples;
  num_validation_samples = prop->num_validation_samples;
  if (prop->model_save_path)
    model_save_path = prop->model_save_path;
  if (prop->model_load_path)
    model_load_path = prop->model_load_path;
  num_epochs = prop->num_epochs;
  total_num_samples =
    (num_training_samples + num_validation_samples) * num_epochs;
  model_config = prop->model_config;

  ml_logd("num_inputs: %d", num_inputs);
  ml_logd("num_labels: %d", num_labels);
  ml_logd("num_training_samples: %d", num_training_samples);
  ml_logd("num_validation_samples: %d", num_validation_samples);
  ml_logd("num_epochs: %d", num_epochs);
  ml_logd("Total number of data to be received: %d", total_num_samples);
  ml_logd("model_config: %s", model_config.c_str());
  ml_logd("model_save_path: %s", model_save_path.c_str());
  ml_logd("model_load_path: %s", model_load_path.c_str());
  ml_logd("<leave>");
}

static int nntrainer_model_destructor(const GstTensorTrainerFramework *fw,
                                      const GstTensorTrainerProperties *prop,
                                      void **private_data) {
  NNTrainer::NNTrainerImpl *nntrainer =
    static_cast<NNTrainer::NNTrainerImpl *>(*private_data);
  UNUSED(fw);
  ml_logd("<called>");

  if (!nntrainer)
    return -1;

  delete nntrainer;
  *private_data = NULL;
  ml_logd("<leave>");

  return 0;
}

static int nntrainer_model_start_training(
  const GstTensorTrainerFramework *fw, const GstTensorTrainerProperties *prop,
  GstTensorTrainerEventNotifier *notifier, void *private_data) {
  NNTrainer::NNTrainerImpl *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerImpl *>(private_data);
  UNUSED(fw);

  ml_logd("<called>");
  if (!nntrainer) {
    ml_loge("Failed get nntrainer");
    return -1;
  }

  if (!notifier) {
    ml_loge("Failed get notify");
    return -1;
  }
  nntrainer->setEventNotifier(notifier);
  nntrainer->setEventNotifierVersion(GST_TENSOR_TRAINER_FRAMEWORK_V1);

  try {
    std::thread train_thread(nntrainer_thread_func, nntrainer);
    train_thread.detach();
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return -1;
  }
  ml_logd("<leave>");
  return 0;
}

static int nntrainer_model_stop_training(const GstTensorTrainerFramework *fw,
                                         const GstTensorTrainerProperties *prop,
                                         void **private_data) {
  NNTrainer::TensorsQueue *train_tensors_queue = nullptr,
                          *valid_tensors_queue = nullptr;
  NNTrainer::NNTrainerImpl *nntrainer =
    static_cast<NNTrainer::NNTrainerImpl *>(*private_data);
  UNUSED(fw);
  ml_logd("<called>");

  if (!nntrainer)
    return -1;

  nntrainer->setStopModelTraining(TRUE);

  ml_logd("<leave>");
  return 0;
}

bool stop_cb(void *user_data) {
  bool *ret = reinterpret_cast<bool *>(user_data);
  ml_logd("<called> %d", *ret);
  return *ret;
}

void epoch_complete_cb(void *user_data) {
  NNTrainer::NNTrainerImpl *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerImpl *>(user_data);

  if (!nntrainer)
    return;
  ml_logd("called epoch_complete_cb!!!");

  /* get current RunStats */
  nntrainer->getRunStats();
  nntrainer->train_tensors_queue->initPushCount();
  nntrainer->valid_tensors_queue->initPushCount();
  /* send event */
  nnstreamer_trainer_notify_event(nntrainer->getEventNotifier(),
                                  TRAINER_EVENT_EPOCH_COMPLETION, NULL);
  return;
}

void NNTrainer::NNTrainerImpl::trainModel() {
  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);
  stop_model_training = false;

  ml_logd("<called>");
  ml_logd("pid[%d], tid[%d]", pid, tid);

  try {
    model->setProperty(
      {"epochs=" + std::to_string(num_epochs), "save_path=" + model_save_path});
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  try {
    model->train({}, stop_cb, &stop_model_training, epoch_complete_cb, this);
    getRunStats();
    ml_loge("[training_loss: %f training_accuracy:%f, validation_loss:%f, "
            "validation_accuracy:%f]",
            train_stats.loss, train_stats.accuracy, valid_stats.loss,
            valid_stats.accuracy);

  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  /* send event */
  nnstreamer_trainer_notify_event(this->notifier,
                                  TRAINER_EVENT_TRAINING_COMPLETION, NULL);
  ml_logd("<leave>");
}

void NNTrainer::NNTrainerImpl::createModel() {
  ml_logd("<called>");
  try {
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  try {
    model->load(model_config, ml::train::ModelFormat::MODEL_FORMAT_INI);
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  try {
    model->compile();
    model->initialize();
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  try {
    if (!model_load_path.empty()) {
      ml_logd("load path : %s", model_load_path.c_str());
      model->load(model_load_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    }
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  ml_logd("<leave>");
}

NNTrainer::NNTrainerImpl::NNTrainerImpl(
  const GstTensorTrainerProperties *prop) {
  ml_logd("<called>");
  setNNStreamerProperties(prop);
  createModel();
  createDataset();
  ml_logd("<leave>");
}
static int nntrainer_model_construct(const GstTensorTrainerFramework *fw,
                                     const GstTensorTrainerProperties *prop,
                                     void **private_data) {
  NNTrainer::NNTrainerImpl *nntrainer =
    static_cast<NNTrainer::NNTrainerImpl *>(*private_data);
  ml_logd("<called>");
  if (nntrainer)
    nntrainer_model_destructor(fw, prop, private_data);
  try {
    nntrainer = new NNTrainer::NNTrainerImpl(prop);
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return -1;
  }

  *private_data = nntrainer;

  ml_logd("<leave>");

  return 0;
}

static int nntrainer_getStatus(const GstTensorTrainerFramework *fw,
                               GstTensorTrainerProperties *prop,
                               void *private_data) {
  NNTrainer::NNTrainerImpl *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerImpl *>(private_data);
  UNUSED(fw);

  if (!nntrainer || !prop) {
    ml_loge("Invalid parameter");
    return -1;
  }

  ml_logd("<called>");

  if (nntrainer->train_tensors_queue && nntrainer->valid_tensors_queue)
    prop->epoch_count = nntrainer->valid_stats.epoch_idx;
  else
    prop->epoch_count = nntrainer->train_stats.epoch_idx;

  prop->training_loss = nntrainer->train_stats.loss;
  prop->training_accuracy = nntrainer->train_stats.accuracy;
  prop->validation_loss = nntrainer->valid_stats.loss;
  prop->validation_accuracy = nntrainer->valid_stats.accuracy;

  ml_loge("%d epochs - [training_loss: %f training_accuracy:%f, "
          "validation_loss:%f, validation_accuracy:%f]",
          prop->epoch_count, prop->training_loss, prop->training_accuracy,
          prop->validation_loss, prop->validation_accuracy);

  return 0;
}

static int nntrainer_getFrameworkInfo(const GstTensorTrainerFramework *fw,
                                      const GstTensorTrainerProperties *prop,
                                      void *private_data,
                                      GstTensorTrainerFrameworkInfo *fw_info) {
  static gchar subplugin_name[] = "nntrainer";
  ml_logd("<called>");
  UNUSED(fw);
  UNUSED(prop);
  UNUSED(private_data);

  fw_info->name = subplugin_name;

  ml_logd("<leave>");
  return 0;
}

static GstTensorTrainerFramework NNS_Trainer_support_nntrainer = {
  .version = GST_TENSOR_TRAINER_FRAMEWORK_V1,
  .create = nntrainer_model_construct,
  .destroy = nntrainer_model_destructor,
  .start = nntrainer_model_start_training,
  .stop = nntrainer_model_stop_training,
  .push_data = nntrainer_model_push_data,
  .getStatus = nntrainer_getStatus,
  .getFrameworkInfo = nntrainer_getFrameworkInfo};

void init_subplugin_nntrainer(void) {
  nnstreamer_trainer_probe(&NNS_Trainer_support_nntrainer);
}

void fini_subplugin_nntrainer(void) {
  nnstreamer_trainer_exit(&NNS_Trainer_support_nntrainer);
}
