GPU_ID=$1


GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_1st.prototxt \
  --weights=mscnn_caltech_train_2nd_iter_25000.caffemodel \
  --gpu=${GPU_ID}  2>&1 | tee log_1st.txt

GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --weights=mscnn_CS_train_1st_iter_10000.caffemodel \
  --gpu=${GPU_ID}  2>&1 | tee log_2nd.txt
