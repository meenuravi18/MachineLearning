	A??ǘQ@A??ǘQ@!A??ǘQ@	????9?J@????9?J@!????9?J@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$A??ǘQ@????????AL?
F%u?@Yh??|??A@*     ??@)      ?=2F
Iterator::ModelM?J??A@!V?D?]R@)?H.?!?6@1?(̸7{G@:Preprocessing2U
Iterator::Model::ParallelMapV2a??+e?)@!z??}:@)a??+e?)@1z??}:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(~??k?)@!??^Xm:@)j?q???)@1V?Vh:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatHP?sע?!???|Mg??)2U0*???1ͅ??g???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?sF???)@!????ċ:@)??A?f??1?گ;5
??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??~j?t??!Aޟ?G	??)??~j?t??1Aޟ?G	??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOv?!̈!q.ǆ?)??_vOv?1̈!q.ǆ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 52.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9????9?J@In9}??G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "      ??!       *      ??!       2	L?
F%u?@L?
F%u?@!L?
F%u?@:      ??!       B      ??!       J	h??|??A@h??|??A@!h??|??A@R      ??!       Z	h??|??A@h??|??A@!h??|??A@b      ??!       JCPU_ONLYY????9?J@b qn9}??G@