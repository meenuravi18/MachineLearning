	vOjM??vOjM??!vOjM??	-?QTF@-?QTF@!-?QTF@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$vOjM??S?!?uq??A??3????YK?46??*	33333s_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ʡE????!?+?$N@G@)?&1???1S??n0ED@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty?&1???!y????A6@)???Mb??1iF?1?2@:Preprocessing2F
Iterator::Model?A`??"??!??jү5@)???QI??1???#?&@:Preprocessing2U
Iterator::Model::ParallelMapV2?HP???!z?We#@)?HP???1z?We#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???Q?~?!?9j???@)???Q?~?1?9j???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$(~??k??!?WeԻS@)_?Q?{?1?Y7?"?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapP??n???!?d?O̢I@)?~j?t?x?1??!Y?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!?????@);?O??nr?1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9-?QTF@I??q]?EX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	S?!?uq??S?!?uq??!S?!?uq??      ??!       "      ??!       *      ??!       2	??3??????3????!??3????:      ??!       B      ??!       J	K?46??K?46??!K?46??R      ??!       Z	K?46??K?46??!K?46??b      ??!       JCPU_ONLYY-?QTF@b q??q]?EX@