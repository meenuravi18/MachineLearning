	??k	? D@??k	? D@!??k	? D@	?W??1A@?W??1A@!?W??1A@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??k	? D@c?ZB>h@Am????r6@Y?[ A??+@*	    @??@2F
Iterator::Model??g??s+@!???J{?X@)Nё\?c+@1????X@:Preprocessing2U
Iterator::Model::ParallelMapV2?U???؟?!?Ѯ?????)?U???؟?1?Ѯ?????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatu????!&?ш???)a??+e??1h??e<??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????o??!ICLĊ{??)????Mb??1ڒ??M???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice%u???!??˙?F??)%u???1??˙?F??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?s????!9	.?ZB??)??~j?t??11w?E????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ZӼ?t?!jj??1???)??ZӼ?t?1jj??1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s9.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?W??1A@I?rԧ/gP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	c?ZB>h@c?ZB>h@!c?ZB>h@      ??!       "      ??!       *      ??!       2	m????r6@m????r6@!m????r6@:      ??!       B      ??!       J	?[ A??+@?[ A??+@!?[ A??+@R      ??!       Z	?[ A??+@?[ A??+@!?[ A??+@b      ??!       JCPU_ONLYY?W??1A@b q?rԧ/gP@