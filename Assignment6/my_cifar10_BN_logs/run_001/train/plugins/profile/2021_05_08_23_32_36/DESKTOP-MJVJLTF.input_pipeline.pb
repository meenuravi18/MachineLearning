	???QI?-@???QI?-@!???QI?-@	'dJ1???'dJ1???!'dJ1???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???QI?-@m???????AM??StD-@Y??Q???*	?????]@2U
Iterator::Model::ParallelMapV2/n????!|????6>@)/n????1|????6>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Ɯ?!? ??8@)?~j?t???1?Τ?љ4@:Preprocessing2F
Iterator::ModelS?!?uq??!?8EG@)HP?sג?11??
??/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!p?2NQ6@)??d?`T??1@?5?Ǻ.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipC??6??!6?Ǻ??J@)/n????1|????6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??ǘ????!B}^??@)??ǘ????1B}^??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?^)?p?!n<ọ'@)	?^)?p?1n<ọ'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9'dJ1???I?7k???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m???????m???????!m???????      ??!       "      ??!       *      ??!       2	M??StD-@M??StD-@!M??StD-@:      ??!       B      ??!       J	??Q?????Q???!??Q???R      ??!       Z	??Q?????Q???!??Q???b      ??!       JCPU_ONLYY'dJ1???b q?7k???X@