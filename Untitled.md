

```python
# 
# This code incorporates lots of the code from the AWS tutorial https://www.youtube.com/watch?v=KCzgR7eQ3PY
# Database generated through the react app from https://github.com/gabehollombe-aws/webcam-sagemaker-inference
# 


# An S3 Bucket Name
data_bucket_name='webcam-s3-uploaderc98413fa7dc14dd280ab0b913fd2b0ec-data'

# A prefix name inside the S3 bucket containing sub-folders of images (one per label class)
dataset_name = 'NewData'
```


```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

role = get_execution_role()
sess = sagemaker.Session()

training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="latest")
```


```python
# Find im2rec in our environment and set up some other vars in our environemnt

base_dir='/tmp'

%env BASE_DIR=$base_dir
%env S3_DATA_BUCKET_NAME = $data_bucket_name
%env DATASET_NAME = $dataset_name

import sys,os

suffix='/mxnet/tools/im2rec.py'
im2rec = list(filter( (lambda x: os.path.isfile(x + suffix )), sys.path))[0] + suffix
%env IM2REC=$im2rec
```

    env: BASE_DIR=/tmp
    env: S3_DATA_BUCKET_NAME=webcam-s3-uploaderc98413fa7dc14dd280ab0b913fd2b0ec-data
    env: DATASET_NAME=NewData
    env: IM2REC=/home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/mxnet/tools/im2rec.py



```python
# Pull our images from S3
!aws s3 sync s3://$S3_DATA_BUCKET_NAME/public/$DATASET_NAME $BASE_DIR/$DATASET_NAME --quiet
```


```bash
%%bash
# Use the IM2REC script to convert our images into RecordIO files

# Clean up our working dir of existing LST and REC files
cd $BASE_DIR
rm *.rec
rm *.lst

# First we need to create two LST files (training and test lists), noting the correct label class for each image
# We'll also save the output of the LST files command, since it includes a list of all of our label classes
echo "Creating LST files"
python $IM2REC --list --recursive --pass-through --test-ratio=0.3 --train-ratio=0.7 $DATASET_NAME $DATASET_NAME > ${DATASET_NAME}_classes

echo "Label classes:"
cat ${DATASET_NAME}_classes

# Then we create RecordIO files from the LST files
echo "Creating RecordIO files"
python $IM2REC --num-thread=4 ${DATASET_NAME}_train.lst $DATASET_NAME
python $IM2REC --num-thread=4 ${DATASET_NAME}_test.lst $DATASET_NAME
ls -lh *.rec
```

    Creating LST files
    Label classes:
    Can 0
    Carton 1
    Test 2
    bottle 3
    Creating RecordIO files
    Creating .rec file from /tmp/NewData_train.lst in /tmp
    time: 0.08516645431518555  count: 0
    Creating .rec file from /tmp/NewData_test.lst in /tmp
    time: 0.004021644592285156  count: 0
    -rw-rw-r-- 1 ec2-user ec2-user 695K Sep  7 22:53 NewData_test.rec
    -rw-rw-r-- 1 ec2-user ec2-user 1.6M Sep  7 22:53 NewData_train.rec


    rm: cannot remove ‘*.rec’: No such file or directory
    rm: cannot remove ‘*.lst’: No such file or directory



```python
# Upload our train and test RecordIO files to S3 in the bucket that our sagemaker session is using
bucket = sess.default_bucket()

s3train_path = 's3://{}/{}/train/'.format(bucket, dataset_name)
s3validation_path = 's3://{}/{}/validation/'.format(bucket, dataset_name)

# Clean up any existing data
!aws s3 rm s3://{bucket}/{dataset_name}/train --recursive
!aws s3 rm s3://{bucket}/{dataset_name}/validation --recursive

# Upload the rec files to the train and validation channels
!aws s3 cp /tmp/{dataset_name}_train.rec $s3train_path
!aws s3 cp /tmp/{dataset_name}_test.rec $s3validation_path
```

    upload: ../../../tmp/NewData_train.rec to s3://sagemaker-us-east-2-425359244402/NewData/train/NewData_train.rec
    upload: ../../../tmp/NewData_test.rec to s3://sagemaker-us-east-2-425359244402/NewData/validation/NewData_test.rec



```python
train_data = sagemaker.session.s3_input(
    s3train_path, 
    distribution='FullyReplicated', 
    content_type='application/x-recordio', 
    s3_data_type='S3Prefix'
)

validation_data = sagemaker.session.s3_input(
    s3validation_path, 
    distribution='FullyReplicated', 
    content_type='application/x-recordio', 
    s3_data_type='S3Prefix'
)

data_channels = {'train': train_data, 'validation': validation_data}
```


```python
s3_output_location = 's3://{}/{}/output'.format(bucket, dataset_name)

image_classifier = sagemaker.estimator.Estimator(
    training_image,
    role, 
    train_instance_count=1, 
    train_instance_type='ml.p2.xlarge',
    output_path=s3_output_location,
    sagemaker_session=sess
)

```


```python
num_classes=! ls -l {base_dir}/{dataset_name} | wc -l
num_classes=int(num_classes[0]) - 1

num_training_samples=! cat {base_dir}/{dataset_name}_train.lst | wc -l
num_training_samples = int(num_training_samples[0])

# Learn more about the Sagemaker built-in Image Classifier hyperparameters here: https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html

# These hyperparameters we won't want to change, as they define things like
# the size of the images we'll be sending for input, the number of training classes we have, etc.
base_hyperparameters=dict(
    use_pretrained_model=1,
    image_shape='3,224,224',
    num_classes=num_classes,
    num_training_samples=num_training_samples,
)

# These are hyperparameters we may want to tune, as they can affect the model training success:
hyperparameters={
    **base_hyperparameters, 
    **dict(
        learning_rate=0.001,
        mini_batch_size=5,
    )
}


image_classifier.set_hyperparameters(**hyperparameters)

hyperparameters
```




    {'use_pretrained_model': 1,
     'image_shape': '3,224,224',
     'num_classes': 4,
     'num_training_samples': 84,
     'learning_rate': 0.001,
     'mini_batch_size': 5}




```python
%%time

import time
now = str(int(time.time()))
training_job_name = 'IC-' + dataset_name.replace('_', '-') + '-' + now

image_classifier.fit(inputs=data_channels, job_name=training_job_name, logs=True)

job = image_classifier.latest_training_job
model_path = f"{base_dir}/{job.name}"

print(f"\n\n Finished training! The model is available for download at: {image_classifier.output_path}/{job.name}/output/model.tar.gz")
```

    2019-09-07 22:53:47 Starting - Starting the training job...
    2019-09-07 22:53:49 Starting - Launching requested ML instances...
    2019-09-07 22:54:44 Starting - Preparing the instances for training.........
    2019-09-07 22:56:02 Downloading - Downloading input data...
    2019-09-07 22:56:39 Training - Downloading the training image......
    2019-09-07 22:57:35 Training - Training image download completed. Training in progress.
    [31mDocker entrypoint called with argument(s): train[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/image_classification/default-input.json: {u'beta_1': 0.9, u'gamma': 0.9, u'beta_2': 0.999, u'optimizer': u'sgd', u'use_pretrained_model': 0, u'eps': 1e-08, u'epochs': 30, u'lr_scheduler_factor': 0.1, u'num_layers': 152, u'image_shape': u'3,224,224', u'precision_dtype': u'float32', u'mini_batch_size': 32, u'weight_decay': 0.0001, u'learning_rate': 0.1, u'momentum': 0}[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'learning_rate': u'0.001', u'num_training_samples': u'84', u'image_shape': u'3,224,224', u'mini_batch_size': u'5', u'use_pretrained_model': u'1', u'num_classes': u'4'}[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] Final configuration: {u'beta_1': 0.9, u'gamma': 0.9, u'beta_2': 0.999, u'optimizer': u'sgd', u'use_pretrained_model': u'1', u'num_classes': u'4', u'eps': 1e-08, u'epochs': 30, u'lr_scheduler_factor': 0.1, u'num_layers': 152, u'image_shape': u'3,224,224', u'precision_dtype': u'float32', u'mini_batch_size': u'5', u'weight_decay': 0.0001, u'learning_rate': u'0.001', u'momentum': 0, u'num_training_samples': u'84'}[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] Searching for .rec files in /opt/ml/input/data/train.[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] Searching for .rec files in /opt/ml/input/data/validation.[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] use_pretrained_model: 1[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] multi_label: 0[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] Using pretrained model for initializing weights and transfer learning.[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] ---- Parameters ----[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] num_layers: 152[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] data type: <type 'numpy.float32'>[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] epochs: 30[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] optimizer: sgd[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] momentum: 0.9[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] weight_decay: 0.0001[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] learning_rate: 0.001[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] num_training_samples: 84[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] mini_batch_size: 5[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] image_shape: 3,224,224[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] num_classes: 4[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] augmentation_type: None[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] kv_store: device[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] checkpoint_frequency not set, will store the best model[0m
    [31m[09/07/2019 22:57:37 INFO 140484553713472] --------------------[0m
    [31m[22:57:37] /opt/brazil-pkg-cache/packages/MXNetECL/MXNetECL-master.883.0/AL2012/generic-flavor/src/src/nnvm/legacy_json_util.cc:209: Loading symbol saved by previous version v0.8.0. Attempting to upgrade...[0m
    [31m[22:57:37] /opt/brazil-pkg-cache/packages/MXNetECL/MXNetECL-master.883.0/AL2012/generic-flavor/src/src/nnvm/legacy_json_util.cc:217: Symbol successfully upgraded![0m
    [31m[09/07/2019 22:57:39 INFO 140484553713472] Setting number of threads: 3[0m
    [31m[22:57:43] /opt/brazil-pkg-cache/packages/MXNetECL/MXNetECL-master.883.0/AL2012/generic-flavor/src/src/operator/nn/./cudnn/./cudnn_algoreg-inl.h:97: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)[0m
    [31m[09/07/2019 22:57:58 INFO 140484553713472] Epoch[0] Train-accuracy=0.375000[0m
    [31m[09/07/2019 22:57:58 INFO 140484553713472] Epoch[0] Time cost=14.863[0m
    [31m[09/07/2019 22:57:59 INFO 140484553713472] Epoch[0] Validation-accuracy=0.771429[0m
    [31m[09/07/2019 22:58:00 INFO 140484553713472] Storing the best model with validation accuracy: 0.771429[0m
    [31m[09/07/2019 22:58:00 INFO 140484553713472] Saved checkpoint to "/opt/ml/model/image-classification-0001.params"[0m
    [31m[09/07/2019 22:58:07 INFO 140484553713472] Epoch[1] Train-accuracy=0.875000[0m
    [31m[09/07/2019 22:58:07 INFO 140484553713472] Epoch[1] Time cost=6.475[0m
    [31m[09/07/2019 22:58:08 INFO 140484553713472] Epoch[1] Validation-accuracy=0.885714[0m
    [31m[09/07/2019 22:58:08 INFO 140484553713472] Storing the best model with validation accuracy: 0.885714[0m
    [31m[09/07/2019 22:58:09 INFO 140484553713472] Saved checkpoint to "/opt/ml/model/image-classification-0002.params"[0m
    [31m[09/07/2019 22:58:15 INFO 140484553713472] Epoch[2] Train-accuracy=0.912500[0m
    [31m[09/07/2019 22:58:15 INFO 140484553713472] Epoch[2] Time cost=6.355[0m
    [31m[09/07/2019 22:58:16 INFO 140484553713472] Epoch[2] Validation-accuracy=0.971429[0m
    [31m[09/07/2019 22:58:17 INFO 140484553713472] Storing the best model with validation accuracy: 0.971429[0m
    [31m[09/07/2019 22:58:17 INFO 140484553713472] Saved checkpoint to "/opt/ml/model/image-classification-0003.params"[0m
    [31m[09/07/2019 22:58:23 INFO 140484553713472] Epoch[3] Train-accuracy=0.975000[0m
    [31m[09/07/2019 22:58:23 INFO 140484553713472] Epoch[3] Time cost=6.343[0m
    [31m[09/07/2019 22:58:25 INFO 140484553713472] Epoch[3] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 22:58:32 INFO 140484553713472] Epoch[4] Train-accuracy=0.937500[0m
    [31m[09/07/2019 22:58:32 INFO 140484553713472] Epoch[4] Time cost=6.460[0m
    [31m[09/07/2019 22:58:33 INFO 140484553713472] Epoch[4] Validation-accuracy=0.900000[0m
    [31m[09/07/2019 22:58:40 INFO 140484553713472] Epoch[5] Train-accuracy=0.887500[0m
    [31m[09/07/2019 22:58:40 INFO 140484553713472] Epoch[5] Time cost=6.357[0m
    [31m[09/07/2019 22:58:41 INFO 140484553713472] Epoch[5] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 22:58:49 INFO 140484553713472] Epoch[6] Train-accuracy=0.987500[0m
    [31m[09/07/2019 22:58:49 INFO 140484553713472] Epoch[6] Time cost=6.374[0m
    [31m[09/07/2019 22:58:50 INFO 140484553713472] Epoch[6] Validation-accuracy=0.885714[0m
    [31m[09/07/2019 22:58:57 INFO 140484553713472] Epoch[7] Train-accuracy=0.962500[0m
    [31m[09/07/2019 22:58:57 INFO 140484553713472] Epoch[7] Time cost=6.402[0m
    [31m[09/07/2019 22:58:58 INFO 140484553713472] Epoch[7] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 22:59:05 INFO 140484553713472] Epoch[8] Train-accuracy=0.987500[0m
    [31m[09/07/2019 22:59:05 INFO 140484553713472] Epoch[8] Time cost=6.343[0m
    [31m[09/07/2019 22:59:06 INFO 140484553713472] Epoch[8] Validation-accuracy=0.971429[0m
    [31m[09/07/2019 22:59:13 INFO 140484553713472] Epoch[9] Train-accuracy=0.962500[0m
    [31m[09/07/2019 22:59:13 INFO 140484553713472] Epoch[9] Time cost=6.475[0m
    [31m[09/07/2019 22:59:15 INFO 140484553713472] Epoch[9] Validation-accuracy=0.900000[0m
    [31m[09/07/2019 22:59:22 INFO 140484553713472] Epoch[10] Train-accuracy=1.000000[0m
    [31m[09/07/2019 22:59:22 INFO 140484553713472] Epoch[10] Time cost=6.476[0m
    [31m[09/07/2019 22:59:23 INFO 140484553713472] Epoch[10] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 22:59:30 INFO 140484553713472] Epoch[11] Train-accuracy=1.000000[0m
    [31m[09/07/2019 22:59:30 INFO 140484553713472] Epoch[11] Time cost=6.360[0m
    [31m[09/07/2019 22:59:31 INFO 140484553713472] Epoch[11] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 22:59:38 INFO 140484553713472] Epoch[12] Train-accuracy=0.962500[0m
    [31m[09/07/2019 22:59:38 INFO 140484553713472] Epoch[12] Time cost=6.376[0m
    [31m[09/07/2019 22:59:40 INFO 140484553713472] Epoch[12] Validation-accuracy=1.000000[0m
    [31m[09/07/2019 22:59:40 INFO 140484553713472] Storing the best model with validation accuracy: 1.000000[0m
    [31m[09/07/2019 22:59:41 INFO 140484553713472] Saved checkpoint to "/opt/ml/model/image-classification-0013.params"[0m
    [31m[09/07/2019 22:59:47 INFO 140484553713472] Epoch[13] Train-accuracy=1.000000[0m
    [31m[09/07/2019 22:59:47 INFO 140484553713472] Epoch[13] Time cost=6.367[0m
    [31m[09/07/2019 22:59:48 INFO 140484553713472] Epoch[13] Validation-accuracy=0.971429[0m
    [31m[09/07/2019 22:59:56 INFO 140484553713472] Epoch[14] Train-accuracy=1.000000[0m
    [31m[09/07/2019 22:59:56 INFO 140484553713472] Epoch[14] Time cost=6.620[0m
    [31m[09/07/2019 22:59:57 INFO 140484553713472] Epoch[14] Validation-accuracy=0.950000[0m
    [31m[09/07/2019 23:00:04 INFO 140484553713472] Epoch[15] Train-accuracy=0.975000[0m
    [31m[09/07/2019 23:00:04 INFO 140484553713472] Epoch[15] Time cost=6.384[0m
    [31m[09/07/2019 23:00:05 INFO 140484553713472] Epoch[15] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 23:00:12 INFO 140484553713472] Epoch[16] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:00:12 INFO 140484553713472] Epoch[16] Time cost=6.371[0m
    [31m[09/07/2019 23:00:13 INFO 140484553713472] Epoch[16] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 23:00:21 INFO 140484553713472] Epoch[17] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:00:21 INFO 140484553713472] Epoch[17] Time cost=6.366[0m
    [31m[09/07/2019 23:00:22 INFO 140484553713472] Epoch[17] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 23:00:29 INFO 140484553713472] Epoch[18] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:00:29 INFO 140484553713472] Epoch[18] Time cost=6.357[0m
    [31m[09/07/2019 23:00:30 INFO 140484553713472] Epoch[18] Validation-accuracy=0.914286[0m
    [31m[09/07/2019 23:00:37 INFO 140484553713472] Epoch[19] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:00:37 INFO 140484553713472] Epoch[19] Time cost=6.504[0m
    [31m[09/07/2019 23:00:38 INFO 140484553713472] Epoch[19] Validation-accuracy=0.925000[0m
    [31m[09/07/2019 23:00:46 INFO 140484553713472] Epoch[20] Train-accuracy=0.987500[0m
    [31m[09/07/2019 23:00:46 INFO 140484553713472] Epoch[20] Time cost=6.375[0m
    [31m[09/07/2019 23:00:47 INFO 140484553713472] Epoch[20] Validation-accuracy=0.914286[0m
    [31m[09/07/2019 23:00:54 INFO 140484553713472] Epoch[21] Train-accuracy=0.975000[0m
    [31m[09/07/2019 23:00:54 INFO 140484553713472] Epoch[21] Time cost=6.384[0m
    [31m[09/07/2019 23:00:55 INFO 140484553713472] Epoch[21] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 23:01:02 INFO 140484553713472] Epoch[22] Train-accuracy=0.950000[0m
    [31m[09/07/2019 23:01:02 INFO 140484553713472] Epoch[22] Time cost=6.402[0m
    [31m[09/07/2019 23:01:03 INFO 140484553713472] Epoch[22] Validation-accuracy=1.000000[0m
    [31m[09/07/2019 23:01:11 INFO 140484553713472] Epoch[23] Train-accuracy=0.962500[0m
    [31m[09/07/2019 23:01:11 INFO 140484553713472] Epoch[23] Time cost=6.465[0m
    [31m[09/07/2019 23:01:12 INFO 140484553713472] Epoch[23] Validation-accuracy=0.942857[0m
    [31m[09/07/2019 23:01:19 INFO 140484553713472] Epoch[24] Train-accuracy=0.962500[0m
    [31m[09/07/2019 23:01:19 INFO 140484553713472] Epoch[24] Time cost=6.491[0m
    [31m[09/07/2019 23:01:20 INFO 140484553713472] Epoch[24] Validation-accuracy=0.950000[0m
    [31m[09/07/2019 23:01:27 INFO 140484553713472] Epoch[25] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:01:27 INFO 140484553713472] Epoch[25] Time cost=6.375[0m
    [31m[09/07/2019 23:01:28 INFO 140484553713472] Epoch[25] Validation-accuracy=0.971429[0m
    [31m[09/07/2019 23:01:36 INFO 140484553713472] Epoch[26] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:01:36 INFO 140484553713472] Epoch[26] Time cost=6.388[0m
    [31m[09/07/2019 23:01:37 INFO 140484553713472] Epoch[26] Validation-accuracy=0.971429[0m
    [31m[09/07/2019 23:01:44 INFO 140484553713472] Epoch[27] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:01:44 INFO 140484553713472] Epoch[27] Time cost=6.501[0m
    [31m[09/07/2019 23:01:45 INFO 140484553713472] Epoch[27] Validation-accuracy=1.000000[0m
    [31m[09/07/2019 23:01:52 INFO 140484553713472] Epoch[28] Train-accuracy=0.987500[0m
    [31m[09/07/2019 23:01:52 INFO 140484553713472] Epoch[28] Time cost=6.372[0m
    [31m[09/07/2019 23:01:53 INFO 140484553713472] Epoch[28] Validation-accuracy=0.971429[0m
    [31m[09/07/2019 23:02:01 INFO 140484553713472] Epoch[29] Train-accuracy=1.000000[0m
    [31m[09/07/2019 23:02:01 INFO 140484553713472] Epoch[29] Time cost=6.482[0m
    [31m[09/07/2019 23:02:02 INFO 140484553713472] Epoch[29] Validation-accuracy=0.950000[0m
    
    2019-09-07 23:02:38 Uploading - Uploading generated training model
    2019-09-07 23:03:10 Completed - Training job completed
    Training seconds: 428
    Billable seconds: 428
    
    
     Finished training! The model is available for download at: s3://sagemaker-us-east-2-425359244402/NewData/output/IC-NewData-1567896827/output/model.tar.gz
    CPU times: user 1.22 s, sys: 60 ms, total: 1.28 s
    Wall time: 9min 45s



```python
%%time
# Deploying a model to an endpoint takes a few minutes to complete

deployed_endpoint = image_classifier.deploy(
    initial_instance_count = 1,
    instance_type = 'ml.t2.medium'
)
```

    ------------------------------------------------------------------------------------------------------------------------------------------!CPU times: user 805 ms, sys: 33.2 ms, total: 839 ms
    Wall time: 11min 37s



```python
import json
import numpy as np
import os

def classify_deployed(file_name, classes):
    payload = None
    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)

    deployed_endpoint.content_type = 'application/x-image'
    result = json.loads(deployed_endpoint.predict(payload))
    best_prob_index = np.argmax(result)        
    print(dataset_name, result)

def classify_frame(f, classes): 
  
    payload = f.read()
    payload = bytearray(payload)

    deployed_endpoint.content_type = 'application/x-image'
    result = json.loads(deployed_endpoint.predict(payload))
    best_prob_index = np.argmax(result)        
    print(dataset_name, result)


```


```python
import time 
while (True): 
    classify_deployed('picture.jpg', dataset_name) 
    !git init
    !git clean -d -f
    !git pull https://github.com/alexander-edwards/PennPics
    time.sleep(2) 
```

    NewData [0.1855064183473587, 0.08078261464834213, 0.7121128439903259, 0.0215982086956501]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Removing .ipynb_checkpoints/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    Removing Untitled.ipynb
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating a217bb0..765ddaf
    Fast-forward
     picture.jpg | Bin [31m248616[m -> [32m209447[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.03264899551868439, 0.01923472434282303, 0.9451702237129211, 0.0029461118392646313]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.03264899551868439, 0.01923472434282303, 0.9451702237129211, 0.0029461118392646313]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.03264899551868439, 0.01923472434282303, 0.9451702237129211, 0.0029461118392646313]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.03264899551868439, 0.01923472434282303, 0.9451702237129211, 0.0029461118392646313]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Removing .ipynb_checkpoints/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    Removing Untitled.ipynb
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating 765ddaf..e4719db
    Fast-forward
     picture.jpg | Bin [31m209447[m -> [32m217055[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.003956971690058708, 0.0009295368799939752, 0.9948884844779968, 0.00022502239153254777]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating e4719db..38149a0
    Fast-forward
     picture.jpg | Bin [31m217055[m -> [32m206120[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.001740843988955021, 0.001002130564302206, 0.9971152544021606, 0.00014181896403897554]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.001740843988955021, 0.001002130564302206, 0.9971152544021606, 0.00014181896403897554]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating 38149a0..2e09bca
    Fast-forward
     picture.jpg | Bin [31m206120[m -> [32m214387[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.00599432410672307, 0.0023140767589211464, 0.991477906703949, 0.0002137235860573128]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.00599432410672307, 0.0023140767589211464, 0.991477906703949, 0.0002137235860573128]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating 2e09bca..b45c90a
    Fast-forward
     picture.jpg | Bin [31m214387[m -> [32m212434[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.005866600666195154, 0.001942352973856032, 0.9920176267623901, 0.000173462278326042]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating b45c90a..f243916
    Fast-forward
     picture.jpg | Bin [31m212434[m -> [32m225463[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.12057604640722275, 0.6197134256362915, 0.03494824096560478, 0.22476224601268768]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.12057604640722275, 0.6197134256362915, 0.03494824096560478, 0.22476224601268768]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating f243916..bd564dd
    Fast-forward
     picture.jpg | Bin [31m225463[m -> [32m224170[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.09246131032705307, 0.03269578516483307, 0.003491438925266266, 0.8713514804840088]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.09246131032705307, 0.03269578516483307, 0.003491438925266266, 0.8713514804840088]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating bd564dd..025592c
    Fast-forward
     picture.jpg | Bin [31m224170[m -> [32m226872[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.011249549686908722, 0.030816497281193733, 0.003201581072062254, 0.954732358455658]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating 025592c..6d9c89c
    Fast-forward
     picture.jpg | Bin [31m226872[m -> [32m202662[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.12406888604164124, 0.7540344595909119, 0.08939157426357269, 0.03250505402684212]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.12406888604164124, 0.7540344595909119, 0.08939157426357269, 0.03250505402684212]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 5, done.[K
    remote: Counting objects: 100% (5/5), done.[K
    remote: Compressing objects: 100% (2/2), done.[K
    remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (3/3), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating 6d9c89c..5a25b33
    Fast-forward
     picture.jpg | Bin [31m202662[m -> [32m219964[m bytes
     1 file changed, 0 insertions(+), 0 deletions(-)
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Removing .ipynb_checkpoints/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    Removing Untitled.ipynb
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Removing .ipynb_checkpoints/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    Removing Untitled.ipynb
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.
    NewData [0.012594997882843018, 0.00860458705574274, 0.9781368374824524, 0.0006635721074417233]
    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Already up-to-date.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-225-ba9b9e7597f4> in <module>()
          5     get_ipython().system('git clean -d -f')
          6     get_ipython().system('git pull https://github.com/alexander-edwards/PennPics')
    ----> 7     time.sleep(2)
    

    KeyboardInterrupt: 



```python

```

    asdfasdfasdfas.jpg  cam.py     PennPics     PLASE.jpg  Untitled Folder
    asdfasd.jpg	    fffff.jpg  picture.jpg  test.jpg   Untitled.ipynb



```python

```


```python

```


```python

```


```python
!ls ../../../
```

    bin	dev   include  local	   mnt	 root  selinux	tmp
    boot	etc   lib      lost+found  opt	 run   srv	usr
    cgroup	home  lib64    media	   proc  sbin  sys	var



```python
!ls 
```

    asdfasdfasdfas.jpg  cam.py     PennPics     PLASE.jpg  Untitled Folder
    asdfasd.jpg	    fffff.jpg  picture.jpg  test.jpg



```python
!git init
!git clean -d -f
!git pull https://github.com/alexander-edwards/PennPics
```

    Reinitialized existing Git repository in /home/ec2-user/SageMaker/.git/
    Skipping repository PennPics/
    Skipping repository Untitled Folder/Untitled Folder/PennPics
    remote: Enumerating objects: 17, done.[K
    remote: Counting objects: 100% (17/17), done.[K
    remote: Compressing objects: 100% (15/15), done.[K
    remote: Total 16 (delta 5), reused 12 (delta 1), pack-reused 0[K
    Unpacking objects: 100% (16/16), done.
    From https://github.com/alexander-edwards/PennPics
     * branch            HEAD       -> FETCH_HEAD
    Updating fcff815..a217bb0
    Fast-forward
     cam.py      |  28 [32m++++++++++++++++++++++++++++[m
     picture.jpg | Bin [31m0[m -> [32m248616[m bytes
     2 files changed, 28 insertions(+)
     create mode 100644 cam.py
     create mode 100644 picture.jpg



```python

```


```python
!sudo pip3 install pyttsx --user
import pyttsx
engine = pyttsx.init()
engine.say('Good morning.')
engine.runAndWait()
```

    Requirement already satisfied: pyttsx in /usr/local/lib/python3.6/site-packages (1.1)
    [33mYou are using pip version 19.0.2, however version 19.2.3 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-232-cc7172034e6c> in <module>()
          1 get_ipython().system('sudo pip3 install pyttsx --user')
    ----> 2 import pyttsx
          3 engine = pyttsx.init()
          4 engine.say('Good morning.')
          5 engine.runAndWait()


    ModuleNotFoundError: No module named 'pyttsx'



```python
!pip install gtts
```

    Collecting gtts
      Downloading https://files.pythonhosted.org/packages/02/0b/e19dd65623e34954fb6793765ad1c6185a669a33e6a6245939e97abeaaca/gTTS-2.0.4-py3-none-any.whl
    Requirement already satisfied: six in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gtts) (1.11.0)
    Collecting gtts-token>=1.1.3 (from gtts)
      Downloading https://files.pythonhosted.org/packages/e7/25/ca6e9cd3275bfc3097fe6b06cc31db6d3dfaf32e032e0f73fead9c9a03ce/gTTS-token-1.1.3.tar.gz
    Requirement already satisfied: click in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gtts) (6.7)
    Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gtts) (2.20.0)
    Requirement already satisfied: beautifulsoup4 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from gtts) (4.6.0)
    Requirement already satisfied: idna<2.8,>=2.5 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from requests->gtts) (2.6)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from requests->gtts) (1.23)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from requests->gtts) (2019.6.16)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from requests->gtts) (3.0.4)
    Building wheels for collected packages: gtts-token
      Building wheel for gtts-token (setup.py) ... [?25ldone
    [?25h  Created wheel for gtts-token: filename=gTTS_token-1.1.3-cp36-none-any.whl size=3273 sha256=a7280a557e90bd533c70631014a905b556a333688ed248e7dac0516e7396a3f7
      Stored in directory: /home/ec2-user/.cache/pip/wheels/dd/11/61/33f7e51bf545e910552b2255eead2a7cd8ef54064b46dceb34
    Successfully built gtts-token
    Installing collected packages: gtts-token, gtts
    Successfully installed gtts-2.0.4 gtts-token-1.1.3



```python
from gtts import gTTS
import os 
import pyglet

a = gTTS("hello", lang = 'en')
a.save('a.mp3')
os.system("a.mp3") 

music = pyglet.media.load('a.mp3', streaming=True)
music.play()



```




    <pyglet.media.player.Player at 0x7fa48e5145f8>




```python
!pip3 install pyglet
```

    Collecting pyglet
    [?25l  Downloading https://files.pythonhosted.org/packages/71/0b/73f209f2b367685302c381284a4b57c2f5d87b9002ca352ed9ad5953944d/pyglet-1.4.3-py2.py3-none-any.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 3.2MB/s eta 0:00:01
    [?25hRequirement already satisfied: future in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from pyglet) (0.17.1)
    Installing collected packages: pyglet
    Successfully installed pyglet-1.4.3



```python

```
