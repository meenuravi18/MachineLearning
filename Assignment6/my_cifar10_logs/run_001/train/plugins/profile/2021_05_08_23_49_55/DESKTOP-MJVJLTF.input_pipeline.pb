	?\m???1@?\m???1@!?\m???1@	??ݍ??????ݍ????!??ݍ????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?\m???1@`??"????A?C?l?k1@Y???V?/??*??????g@)       =2U
Iterator::Model::ParallelMapV2?E???Ԩ?!???7?V9@)?E???Ԩ?1???7?V9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ڊ?e??!?L?Go?7@)M?J???1? ??m4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\ A?c̭?!???uSh>@)a2U0*???1*u?4@:Preprocessing2F
Iterator::Model?HP???!%C?x?|C@)}гY????1E]t?E+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??ׁsF??!??=??$@)??ׁsF??1??=??$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?<,Ԛ???!ۼ??N@)?{??Pk??1r? A??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?{?!a???@)F%u?{?1a???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??ݍ????I?D?"?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	`??"????`??"????!`??"????      ??!       "      ??!       *      ??!       2	?C?l?k1@?C?l?k1@!?C?l?k1@:      ??!       B      ??!       J	???V?/?????V?/??!???V?/??R      ??!       Z	???V?/?????V?/??!???V?/??b      ??!       JCPU_ONLYY??ݍ????b q?D?"?X@