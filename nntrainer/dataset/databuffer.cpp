/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	databuffer.cpp
 * @date	04 December 2019
 * @brief	This is buffer object to handle big data
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <base_properties.h>
#include <cassert>
#include <climits>
#include <cstring>
#include <databuffer.h>
#include <func_data_producer.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief Props containing buffer size value
 *
 */
class PropsBufferSize : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new props min object with a default value
   *
   * @param value default value
   */
  PropsBufferSize(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "buffer_size"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

constexpr char USER_DATA[] = "user_data";

DataBuffer::DataBuffer(std::unique_ptr<DataProducer> &&producer_) :
  producer(std::move(producer_)), db_props(new Props()), user_data(nullptr) {
  rng.seed(0);
}

DataBuffer::~DataBuffer(){};

std::future<std::shared_ptr<IterationQueue>>
DataBuffer::startFetchWorker(const std::vector<TensorDim> &input_dims,
                             const std::vector<TensorDim> &label_dims,
                             bool shuffle) {
  NNTR_THROW_IF(!producer, std::runtime_error) << "producer does not exist";
  NNTR_THROW_IF(input_dims.empty(), std::runtime_error)
    << "There must be at least one input";

  auto q_size = std::get<PropsBufferSize>(*db_props);
  auto iq = std::make_shared<IterationQueue>(q_size, input_dims, label_dims);
  auto generator = producer->finalize(input_dims, label_dims);
  auto size = producer->size(input_dims, label_dims);
  iq_view = iq;

  class NotifyOnDestruct {
  public:
    NotifyOnDestruct(IterationQueue *iq) : iq(iq) {}
    ~NotifyOnDestruct() {
      try {
        iq->notifyEndOfRequestEmpty();
      } catch (std::exception &e) {
        ml_loge("failed to notify end of request, reason: %s", e.what());
      }
    }

  private:
    IterationQueue *iq = iq;
  };

  /// case of generator
  if (size == DataProducer::SIZE_UNDEFINED) {
    return std::async(std::launch::async, [iq, generator] {
      auto notifier = NotifyOnDestruct(iq.get());
      for (unsigned int i = 0; i < DataProducer::SIZE_UNDEFINED; ++i) {
        /// below loop can be parallelized
        auto sample_view = iq->requestEmptySlot();
        NNTR_THROW_IF(sample_view.isEmpty(), std::runtime_error)
          << "[Databuffer] Cannot fill empty buffer";
        auto &sample = sample_view.get();
        try {
          bool last =
            generator(i, sample.getInputsRef(), sample.getLabelsRef());
          if (last) {
            break;
          }
        } catch (std::exception &e) {
          ml_loge("Fetching sample failed, Error: %s", e.what());
          throw;
        }
      }

      return iq;
    });
  }

  std::vector<unsigned int> idxes_;
  if (shuffle == true) {
    idxes_.resize(size);
    std::iota(idxes_.begin(), idxes_.end(), 0);
    std::shuffle(idxes_.begin(), idxes_.end(), rng);
  }

  return std::async(std::launch::async, [iq, generator, size,
                                         idxes = std::move(idxes_), shuffle] {
    auto notifier = NotifyOnDestruct(iq.get());
    for (unsigned int i = 0; i < size; ++i) {
      /// below loop can be parallelized
      auto sample_view = iq->requestEmptySlot();
      NNTR_THROW_IF(sample_view.isEmpty(), std::runtime_error)
        << "[Databuffer] Cannot fill empty buffer";
      auto &sample = sample_view.get();
      try {
        generator(shuffle ? idxes[i] : i, sample.getInputsRef(),
                  sample.getLabelsRef());
      } catch (std::exception &e) {
        ml_loge("Fetching sample failed, Error: %s", e.what());
        throw;
      }
    }

    return iq;
  });
}

ScopedView<Iteration> DataBuffer::fetch() {
  NNTR_THROW_IF(!producer, std::runtime_error) << "producer does not exist";
  auto iq = iq_view.lock();
  NNTR_THROW_IF(!iq, std::runtime_error)
    << "Cannot fetch, either fetcher is not running or fetcher has ended and "
       "invalidated";
  return iq->requestFilledSlot();
}

std::tuple<DataProducer::Generator /** generator */, unsigned int /** size */>
DataBuffer::getGenerator(const std::vector<TensorDim> &input_dims,
                         const std::vector<TensorDim> &label_dims) {
  NNTR_THROW_IF(!producer, std::invalid_argument) << "producer does not exist";
  return {producer->finalize(input_dims, label_dims),
          producer->size(input_dims, label_dims)};
}

void DataBuffer::displayProgress(const int count, float loss) {
  int barWidth = 20;
  /** this is temporary measure, will be getting this as an argument */
  int batch_size = 1;
  int samples_per_epoch = 0;

  std::stringstream ssInt;
  ssInt << count * batch_size;

  std::string str = ssInt.str();
  int len = str.size();

  if (samples_per_epoch == 0) {
    int pad_left = (barWidth - len) / 2;
    int pad_right = barWidth - pad_left - len;
    std::string out_str =
      std::string(pad_left, ' ') + str + std::string(pad_right, ' ');
    std::cout << " [ ";
    std::cout << out_str;
    std::cout << " ] "
              << " ( Training Loss: " << loss << " )\r";
  } else {
    float progress;
    if (batch_size > samples_per_epoch)
      progress = 1.0;
    else
      progress = (((float)(count * batch_size)) / (float)samples_per_epoch);

    int pos = (int)(barWidth * progress);
    std::cout << " [ ";
    for (int l = 0; l < barWidth; ++l) {
      if (l <= pos)
        std::cout << "=";
      else
        std::cout << " ";
    }
    std::cout << " ] " << int(progress * 100.0) << "% ( Training Loss: " << loss
              << " )\r";
  }

  std::cout.flush();
}

void DataBuffer::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, *db_props);
  if (producer) {
    producer->setProperty(left);
  } else {
    NNTR_THROW_IF(!left.empty(), std::invalid_argument)
      << "[DataBuffer] Failed to set property";
  }
}

const std::string DataBuffer::getType() const {
  NNTR_THROW_IF(!producer, std::invalid_argument) << "producer is empty";
  return producer->getType();
}

void DataBuffer::exportTo(Exporter &exporter,
                          const ml::train::ExportMethods &method) const {
  if (producer) {
    producer->exportTo(exporter, method);
  }
  exporter.saveResult(*db_props, method, this);
}

bool DataBuffer::isSerializable(const ml::train::ExportMethods &method) const {
  if (method != ml::train::ExportMethods::METHOD_STRINGVECTOR) {
    return false;
  }
  if (!producer) {
    return false;
  }

  /// @todo this should be query from producer->isSerializable
  if (producer->getType() == FuncDataProducer::type) {
    return false;
  }
  return true;
}
} /* namespace nntrainer */
