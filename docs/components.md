## Components

### Supported Layers

This component defines layers which consist of a neural network model. Layers have their own properties to be set.

| Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
| conv1d | Conv1DLayer | Convolution 1-Dimentional Layer |
| conv2d | Conv2DLayer |Convolution 2-Dimentional Layer |
| pooling2d | Pooling2DLayer |Pooling 2-Dimentional Layer. Support average / max / global average / global max pooling |
| flatten | FlattenLayer | Flatten layer |
| fully_connected | FullyConnectedLayer | Fully connected layer |
| input | InputLayer | Input Layer.  This is not always required. |
| batch_normalization | BatchNormalizationLayer | Batch normalization layer |
| layer_normalization | LayerNormalizationLayer | Layer normalization layer |
| activation | ActivationLayer | Set by layer property |
| addition | AdditionLayer | Add input input layers |
| attention | AttentionLayer | Attenstion layer |
| centroid_knn | CentroidKNN | Centroid K-nearest neighbor layer |
| concat | ConcatLayer | Concatenate input layers |
| multiout | MultiOutLayer | Multi-Output Layer |
| backbone_nnstreamer | NNStreamerLayer | Encapsulate NNStreamer layer |
| backbone_tflite | TfLiteLayer | Encapsulate tflite as a layer |
| permute | PermuteLayer | Permute layer for transpose |
| preprocess_flip | PreprocessFlipLayer | Preprocess random flip layer |
| preprocess_l2norm | PreprocessL2NormLayer | Preprocess simple l2norm layer to normalize |
| preprocess_translate | PreprocessTranslateLayer | Preprocess translate layer |
| reshape | ReshapeLayer | Reshape tensor dimension layer |
| split | SplitLayer | Split layer |
| dropout | DropOutLayer | Dropout Layer |
| embedding | EmbeddingLayer | Embedding Layer |
| positional_encoding | PositionalEncodingLayer | Positional Encoding Layer |
| rnn | RNNLayer | Recurrent Layer |
| rnncell | RNNCellLayer | Recurrent Cell Layer |
| gru | GRULayer | Gated Recurrent Unit Layer |
| grucell | GRUCellLayer | Gated Recurrent Unit Cell Layer |
| lstm | LSTMLayer | Long Short-Term Memory Layer |
| lstmcell | LSTMCellLayer | Long Short-Term Memory Cell Layer |
| zoneoutlstmcell | ZoneoutLSTMCellLayer | Zoneout Long Short-Term Memory Cell Layer |
| time_dist | TimeDistLayer | Time distributed Layer |
| multi_head_attention | MultiHeadAttentionLayer | Multi Head Attention Layer |
