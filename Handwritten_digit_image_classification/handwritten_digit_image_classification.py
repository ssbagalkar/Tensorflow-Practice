# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print("Data imported...")


import tensorflow as tf
from pytictoc import TicToc
t = TicToc()


#Set our hyperparameters
learning_rate = 0.01
num_epochs = 20
batch_size = 30
print("All hyperparameters are set...")


# TF graph input
with tf.name_scope("input"):
    x = tf.placeholder("float",[None, 784], name="x-input") # mnist data images of 28*28=784
    y = tf.placeholder("float",[None, 10], name="y-input") # 10 class labels

with tf.name_scope("input-reshape"):
    image_input_reshape = tf.reshape(x,[-1,28,28,1])
    tf.summary.image("input",image_input_reshape,10)


# Create model
# Set weights
with tf.name_scope("weights"):
    W = tf.Variable(tf.zeros([784,10]))

# Set bias
with tf.name_scope("bias"):
    b = tf.Variable(tf.zeros(10))


# Create a model with Wx+b as scope
with tf.name_scope("Wx_plus_b") as scope:
    # Solve the linear equation to get logits
    logits = tf.matmul(x,W)+b

with tf.name_scope("softmax") as scope:
    # Once we get the logits, we can turn these into probabilities using softmax function
    model = tf.nn.softmax(logits)


# add summary operations to visualize the distribution of weights and biases
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)



# Define the cost function
with tf.name_scope("cross_entropy") as scope:
    cost_function = -tf.reduce_sum(tf.multiply(y,tf.log(model)))
    
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function-xentropy",cost_function)



# Train the model
with tf.name_scope("train") as scope:
    # Define the gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Test the model
with tf.name_scope("accuracy"):
    # get predictions
    predictions = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions,"float"))

    # Create a summary for accuracy
    tf.summary.scalar("accuracy",accuracy)

# Merge all summaries
merged_summary_op = tf.summary.merge_all()


# Initialize al variables
init = tf.global_variables_initializer()


# Launch graph
print("start training...")
t.tic()


with tf.Session() as sess:
    sess.run(init)
    # Set the logs writer
    summary_writer = tf.summary.FileWriter('C:\\Users\\saurabh B\\TensorFlow-practice\\Tensorflow-Practice\\Summary_logs', graph=sess.graph)
    
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0.
        batch_count = int(mnist.train.num_examples/batch_size)
        
        # Loop over all batches
        for batch in range(batch_count):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            # Train the model 
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
            # Compute average loss/cost
            avg_cost += sess.run(cost_function,feed_dict={x:batch_x,y:batch_y})/batch_count
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op,feed_dict={x:batch_x,y:batch_y})
            summary_writer.add_summary(summary_str,epoch*batch_count+batch)
            
        # Display logs per iteration step
        if epoch % 5 == 0:
            print("Epoch:", '%02d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
            
    print("Training Complete")    
    t.toc()

    print ("Accuracy:", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))

