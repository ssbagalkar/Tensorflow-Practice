# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print("Data imported...")


import tensorflow as tf




#Set our hyperparameters
learning_rate = 0.01
training_iterations = 30
batch_size = 30
#display_step = 2
print("All hyperparameters are set...")


# TF graph input
x = tf.placeholder("float",[None, 784]) # mnist data images of 28*28=784
y = tf.placeholder("float",[None, 10]) # 10 class labels


# Create model
# Set weights
W = tf.Variable(tf.zeros([784,10]))

# Set bias
b = tf.Variable(tf.zeros(10))


# Create a model with Wx+b as scope
with tf.name_scope("Wx_b") as scope:
    # Solve the linear equation to get logits
    logits = tf.matmul(x,W)+b

    # Once we get the logits, we can turn these into probabilities using softmax function
    model = tf.nn.softmax(logits)


# add summary operations to visulaize the distribution of weights and biases
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)



# Define the cost function
with tf.name_scope("cost_function") as scope:
    cost_function = -tf.reduce_sum(tf.mul(y,tf.log(model)))
    
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function",cost_function)



# Train the model
with tf.name_scope("train") as scope:
    # Define the gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


# Initialize al variables
init = tf.global_variables_initializer()


# Merge all summaries
merged_summary_op = tf.merge_all_summaries()


# Launch graph
print("start training...")
with tf.Session() as sess:
    sess.run(init)
    # Set the logs writer
    summary_writer = tf.summary.FileWriter('C:\\Users\\saurabh B\\TensorFlow-practice\\Tensorflow-Practice\\Handwritten_digit_image classification', graph=sess.graph)
    
    # Training cycle
    for iteration in range(training_iterations):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        # Loop over all batches
        for batch in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            # Train the model 
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
            # Compute average loss/cost
            avg_cost += sess.run(cost_function,feed_dict={x:batch_x,y:batch_y})/total_batch
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op,feed_dict={x:batch_x,y:batch_y})
            summary_writer.add_summary(summary_str,iteration*total_batch*batch)
            
        # Display logs per iteration step
        #if iteration % display_step == 0:
        print ("Iteration:", '%02d' % (iteration), "cost=", "{:.9f}".format(avg_cost))
            
    print("Training Complete")    

    # Test the model
    predictions = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions,"float"))

    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

