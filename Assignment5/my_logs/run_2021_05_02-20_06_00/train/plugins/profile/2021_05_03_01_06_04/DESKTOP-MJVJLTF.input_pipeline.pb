	??\@??\@!??\@	*??6?R	@*??6?R	@!*??6?R	@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??\@??ͪ????Aڬ?\m???Y6<?R???*	33333SZ@2U
Iterator::Model::ParallelMapV2??H?}??!??Y??Y;@)??H?}??1??Y??Y;@:Preprocessing2F
Iterator::Model+??Χ?!G?i?kF@)r??????1??yG"?0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj?t???!ܣv?j4@)r??????1??yG"?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV}??b??!Y:P???7@)??ǘ????1O??N??.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u???!?>???K@)M?O???1?]K??.#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?5?;Nс?!c??3? @)?5?;Nс?1c??3? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!o?!?xPy??@)ŏ1w-!o?1?xPy??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_?Qڛ?!??b???9@)HP?s?b?1@xPy@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9*??6?R	@I_*Hi5X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ͪ??????ͪ????!??ͪ????      ??!       "      ??!       *      ??!       2	ڬ?\m???ڬ?\m???!ڬ?\m???:      ??!       B      ??!       J	6<?R???6<?R???!6<?R???R      ??!       Z	6<?R???6<?R???!6<?R???b      ??!       JCPU_ONLYY*??6?R	@b q_*Hi5X@