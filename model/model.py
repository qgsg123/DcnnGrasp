import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Concatenate, \
    Dropout, Input, Flatten, AveragePooling2D

def Creat_DcnnGrasp(class_num, grasp_num):
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',  # Pre-trained weights on ImageNet dataset
        input_shape=(224, 224, 3),  # Image shape of resized CIFAR10
        include_top=False,  # Do not include the classifier part (we write our own)
    )

    # Base model should not be updated (i.e. we freeze the weights)
    base_model.trainable = False

    # create first one
    inp1 = tf.keras.layers.Input(shape=(224, 224, 3), name='input_1')
    # Inputs needs to be pre-processed for base model
    b1 = tf.keras.layers.Lambda(tf.keras.applications.densenet.preprocess_input)(inp1)
    # Pass inputs in the base model (training false means the model is used in
    # inference mode, which will allow us fine tuning after initial training)
    b1 = base_model(b1, training=False)
    # We need to implement a new classifier. For this need to flatten the output
    # of the base_model
    b1 = tf.keras.layers.Flatten()(b1)
    # This is the actual classifier
    b1 = tf.keras.layers.BatchNormalization(name='cate_b_1')(b1)
    b1 = tf.keras.layers.Dense(256, activation='relu', name='cate_d_1')(b1)
    b1 = tf.keras.layers.Dropout(0.3)(b1)
    b1 = tf.keras.layers.BatchNormalization(name='cate_b_2')(b1)
    b1 = tf.keras.layers.Dense(128, activation='relu', name='cate_d_2')(b1)
    b1 = tf.keras.layers.Dropout(0.3)(b1)
    b1 = tf.keras.layers.BatchNormalization(name='cate_b_3')(b1)
    b1 = tf.keras.layers.Dense(64, activation='relu', name='cate_d_3')(b1)
    b1 = tf.keras.layers.Dropout(0.3)(b1)

    # Output is a dense classifier with softmax activation
    out1 = tf.keras.layers.Dense(class_num, activation='softmax', name='cate_d_4')(b1)

    # create second one
    inp2 = tf.keras.layers.Input(shape=(48, 36, 1), name='input2')
    b2 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), activation='relu', name='grasp_c_1')(inp2)
    b2 = tf.keras.layers.Conv2D(filters=25, kernel_size=(5, 5), strides=(1, 1), activation='relu', name='grasp_c_2')(b2)
    b2 = tf.keras.layers.MaxPool2D((2, 2))(b2)
    b2 = tf.keras.layers.Flatten()(b2)
    # Concatenation along filter dimensions
    # b = np.concatenate(axis=0)((b1, b2))

    # attention_layer = Attention_layer()
    # b1 = attention_layer(b1)

    b = Concatenate(axis=-1)([b1, b2])
    # b = tf.keras.layers.Flatten()(b)
    b = tf.keras.layers.Dense(128, activation='relu', name='grasp_d_1')(b)
    b = tf.keras.layers.Dense(64, activation='relu', name='grasp_d_2')(b)
    out = tf.keras.layers.Dense(grasp_num, activation='softmax', name='grasp_d_3')(b)

    # Define the model
    # model = tf.keras.models.Model(Concatenate(axis=-1)([inp1, inp2]), out)
    final_out = tf.concat([out1, out], axis=1)
    model = tf.keras.models.Model([inp1, inp2], final_out)

    return model