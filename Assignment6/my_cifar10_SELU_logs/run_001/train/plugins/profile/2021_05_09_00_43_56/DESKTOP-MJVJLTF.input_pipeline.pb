	,e?XW3@,e?XW3@!,e?XW3@	??ʒ??????ʒ????!??ʒ????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$,e?XW3@8gDio??A(???2@YEGr????*?????9[@)      =2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt$???~??!?YÁاH@)EGr????1??7?qE@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ?o_Ι?!J???#7@)?g??s???1?S??w3@:Preprocessing2F
Iterator::ModelHP?s??!??U?I5@)?(??0??1?rW???&@:Preprocessing2U
Iterator::Model::ParallelMapV2?g??s???!?S??w#@)?g??s???1?S??w#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj?t???!????-?S@)?ZӼ?}?1;?yz7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicey?&1?|?!?5[|/?@)y?&1?|?1?5[|/?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!?όib@)????Mbp?1?όib@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??ʒ????I?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8gDio??8gDio??!8gDio??      ??!       "      ??!       *      ??!       2	(???2@(???2@!(???2@:      ??!       B      ??!       J	EGr????EGr????!EGr????R      ??!       Z	EGr????EGr????!EGr????b      ??!       JCPU_ONLYY??ʒ????b q?????X@