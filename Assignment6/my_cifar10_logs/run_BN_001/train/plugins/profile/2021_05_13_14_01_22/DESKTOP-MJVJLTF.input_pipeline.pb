	?~?:p?7@?~?:p?7@!?~?:p?7@	??a 8????a 8??!??a 8??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?~?:p?7@?	?c??A?W[??|7@Y?Fx$??*	33333?_@2U
Iterator::Model::ParallelMapV2P?s???!?%^k?;@)P?s???1?%^k?;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??y?):??!ח?t$?;@)u????1rL?[e7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??????!`?Xޫ-8@)Zd;?O???1X?&?2?1@:Preprocessing2F
Iterator::Model	?c???!O??J;?D@)e?X???1???T?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipU???N@??!?;??kM@)?5?;Nс?1??W[?:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicevq?-??! P????@)vq?-??1 P????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?+e?Xw?!????"?@)?+e?Xw?1????"?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??a 8??IP?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?	?c???	?c??!?	?c??      ??!       "      ??!       *      ??!       2	?W[??|7@?W[??|7@!?W[??|7@:      ??!       B      ??!       J	?Fx$???Fx$??!?Fx$??R      ??!       Z	?Fx$???Fx$??!?Fx$??b      ??!       JCPU_ONLYY??a 8??b qP?????X@