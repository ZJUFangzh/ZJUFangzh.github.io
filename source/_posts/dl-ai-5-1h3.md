---
title: 'DeepLearning.ai作业:(5-1)-- 循环神经网络（Recurrent Neural Networks）（3）'
id: dl-ai-5-1h3
tags:
  - dl.ai
  - homework
categories:
  - AI
  - Deep Learning
date: 2018-10-18 16:20:36
---


![](http://ww1.sinaimg.cn/large/d40b6c29gy1fvrl8dyhm4j218w0nstdc.jpg)

第三个作业是用LSTM来生成爵士乐。

<!--more-->

# Part3:Improvise a Jazz Solo with an LSTM Network



我们已经对音乐数据做了预处理，以”values”来表示。可以非正式地将每个”value”看作一个音符，它包含音高和持续时间。 例如，如果您按下特定钢琴键0.5秒，那么您刚刚弹奏了一个音符。 在音乐理论中，”value” 实际上比这更复杂。 特别是，它还捕获了同时播放多个音符所需的信息。 例如，在播放音乐作品时，可以同时按下两个钢琴键（同时播放多个音符生成所谓的“和弦”）。 但是这里我们不需要关系音乐理论的细节。对于这个作业，你需要知道的是，我们获得一个”values”的数据集，并将学习一个RNN模型来生成一个序列的”values”。

我们的音乐生成系统将使用78个独特的值。



- X: 这是一个（m，Tx，78）维数组。 m 表示样本数量，Tx 表示时间步(也即序列的长度)，在每个时间步，输入是78个不同的可能值之一，表示为一个one-hot向量。 因此，例如，X [i，t，：]是表示第i个示例在时间t的值的one-hot向量。
- Y: 与X基本相同，但向左（向前）移动了一步。 与恐龙分配类似，使用先前值预测下一个值，所以我们的序列模型将尝试预测给定的x⟨t⟩。 但是，Y中的数据被重新排序为维（Ty，m，78），其中Ty = Tx。 这种格式使得稍后进入LSTM更方便。
- n_value: 数据集中独立”value”的个数，这里是78
- indices_values: python 字典：key 是0-77，value 是特定音符



模型结构如下：

![](http://ww1.sinaimg.cn/large/d40b6c29ly1fwcgolo43mj21sm0ryq6x.jpg)

这里用了3个keras函数来定义：

```python
reshapor = Reshape((1, 78))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D
```



```python
# GRADED FUNCTION: djmodel

def djmodel(Tx, n_a, n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 

    Returns:
    model -- a keras model with the 
    """

    # Define the input of your model with a shape 
    X = Input(shape=(Tx, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    ### START CODE HERE ### 
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []

    # Step 2: Loop
    for t in range(Tx):

        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda x: X[:,t,:])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)

    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model

```

```
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
```

```
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

```
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
```

```
model.fit([X, a0, c0], list(Y), epochs=100)
```



## 生成音乐的模型

```python
# GRADED FUNCTION: music_inference_model

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        x = Lambda(one_hot)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model
```

```
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)
```

```
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
```

```python
# GRADED FUNCTION: predict_and_sample

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=x_initializer.shape[-1])
    ### END CODE HERE ###
    
    return results, indices
```

```
out_stream = generate_music(inference_model)
```

