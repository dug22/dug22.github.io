## Text Generation Using Recurrent Neural Networks (RNN) 

## Introduction

<p align="justify">
Recurrent Neural Networks (RNNs) are a type of neural network designed to process and handle sequence prediction tasks. Sequence data such as time series data, natural language, and other similar datasets. 
RNNs process sequence data by utilizing previous outputs as inputs while maintaining hidden states to capture context over time. Unlike Feedforward models RNNs pretty much have the capability to maintain 
memory of previous inputs. We'll be going over the fundamentals of how RNNs work, along with building our own RNN model to predict the next set of characters of a phrase based on the start of a list of 
characters in Python using Google Colab. 
</p>


## Examples of RNNs Handling Sequential Data


Let's assume we have a couple of phrases we want our RNN model to remember. Below, we have two phrases we want to feed our model.
  <ul>
    <li>"The color of a ripe tomato is ___"</li>
    <li>"The color of the ocean is deep ____"</li>
  </ul>

<p align="justify">
I would assume your answer for question one was "red" and question two was "blue". You likely arrived at these answers effortlessly because our brains excel at identifying missing words in context. We understand the 
missing word by considering the earlier part of the sentence. Instead of focusing on just one word in isolation, we retain and use the prior information to determine the correct answer.
</p>

 <ul>
    <li>A simple RNN would look something like this:</li>
    <ul>
      <li><b>x</b> represents givens inputs</li>
      <li><b>A</b> represents its hidden layers with loops in them allowing for them to persist</li>
      <li><b>h</b> represents given outputs</li>
    </ul>
  </ul>

<p align="center">
    <img src="https://camo.githubusercontent.com/0c8708d6d219e7a52a0dc3044b66ea03cb1738167aa6d1eaaccd5cd2a5f9f8e4/68747470733a2f2f656e637279707465642d74626e302e677374617469632e636f6d2f696d616765733f713d74626e3a414e643947635462336d35436577796e66624336354b4e626a6d4f643044306551346942724e43326e676f7a4e5f325a3469304976392d4762432d486c35526c466d5a6a325939565a793026757371703d434155" alt="Your Image" style="margin: 20px;">
</p>

* Now when it comes to unfolding this RNN it would look something like this:</li>

<p align="right">
<img src="https://camo.githubusercontent.com/f5d3e18f7b294903a87445539e268b8cbdfe2103f1355e4ba9f69e00af42944b/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f76322f726573697a653a6669743a313430302f302a63314c396a6a6373415361676b5f48752e706e67">
</p>


<p align="justify">
  <ul>
    <li>The diagram above illustrates sequence data with multiple time steps, where each time step incorporates information from all previous steps. We could take the final output in the diagram and pass it to a
simple linear classification layer for word prediction, as shown in the example above. This should give you a basic understanding of how an RNN model processes sequential data.</li>
  </ul>
</p>


  
## Fundamentals of Backpropagation and the Vanishing Gradient Problem

<p align="justify">
Backpropagation is an algorithm designed to detect errors by working backwards from the output nodes to the input nodes. In simple terms, it is the process of fine-tuning a neural network's weights to enhance prediction accuracy. Here is how a neural network 
typically works during the backpropagation process:
   <ol>
      <li>First, the neural network performs a forward pass and makes a prediction.</li>
      <li>Next, it compares the prediction to the true value using a loss function. The loss function outputs an error value, indicating how poorly the neural network performed, on a scale of 1-10.</li>
      <li>Then, it uses this error value to calculate gradients for each node within the network. These gradients are used to adjust the network’s internal weights, allowing it to learn. Larger gradients lead to bigger weight adjustments.</li>
      <li>Here is where things get tricky. During backpropagation, each node calculates its gradient with respect to the effects of the gradients in the previous layer. If the adjustments in the preceding layer are small, the gradients shrink further as they move backward. 
          This causes the gradient values to get smaller as they are propagated from layer to layer.</li>
    </ol>
  

This leads us to the vanishing gradient problem, where gradients become so small that the network fails to learn properly. To visualize this concept, here’s an animated gif demonstrating backpropagation:
</p>
<p align="center">
<img src="https://camo.githubusercontent.com/ce0a37f931f3d00f89d32e6af908827291ec44ebc64f403060f04d5a4c7c5af2/68747470733a2f2f6d616368696e656c6561726e696e676b6e6f776c656467652e61692f77702d636f6e74656e742f75706c6f6164732f323031392f31302f4261636b70726f7061676174696f6e2e676966">
</p>

## Applying Backpropagation to RNNs
Now that you have an understanding of backpropagation and the vanishing gradient problem, let's explore how these concepts apply to RNNs.

<p align="justify">
<ul>
  <li>We can think of each time step as a layer. When applying backpropagation in this context, it is often referred to as "backpropagation through time." Gradients tend to shrink with each time step. Please refer to the model below:</li>
</ul>
</p>


<p align="center">
<img src="https://camo.githubusercontent.com/abd9864d586c43a55b1a2e70a1ac26b6f87cf92d715ef7ca699466035bb6c72a/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a4b753534716d4372795a5642614963366738726a47412e676966">
</p>

  <ul>
  <li> <p align="justify"> The node that is initially last in the diagram will be the first node where backpropagation takes place. As it moves down the list of nodes, the gradient will start becoming smaller and smaller, potentially causing the other nodes to fail to learn.</p></li>
</ul>



## How Developers Address This Issue

<p align="justify">
Even though RNNs are useful, they struggle to retain information over long periods of time, often forgetting what should be remembered. They are effective for short-term tasks, but their limitations prompted developers to address this issue,
and seek improved solutions. Over time, developers have found more effective ways to address the vanishing gradient problem. The creation of models like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) has provided solutions to mitigate the gradient issues inherent in RNNs.
</p>

## Imports

Now that we understand how RNNs work and process sequential data, let's dive into the code. These are the imports we need for our Google Colab Notebook.


~~~python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
~~~~

## Dataset

Next we need to create our given dataset to train our RNN model. Our dataset will consist a unique list of simplified phrases that will be used to train our model.

~~~python
phrases = [
    "oceans are deep blue",
    "tomatoes are ripe red",
    "grass is green",
    "pennies are really shiny",
    "lakes are blue"
]
~~~

## Printing The Length of Each Phrase

Here, we are looping through each phrase and printing the phrase along with the number of characters that phrase contains.

~~~python
length_of_chars = [len(phrase) for phrase in phrases]
for phrase, length in zip(phrases, length_of_chars):
    print(f"Text: '{phrase}' | Length: {length}")
~~~

```
Output:
Text: 'oceans are deep blue' | Length: 20
Text: 'tomatoes are ripe red' | Length: 21
Text: 'grass is green' | Length: 14
Text: 'pennies are really shiny' | Length: 24
Text: 'lakes are blue' | Length: 14
```

## Modifying Our Dataset

Before creating our RNN model or feeding it any of our data, we need to maodify our dataset to be prepared for training.

### Concentenating, Converting the String into a Set of Characters, and Sorting

<p align="justify">
  <ul>
    <li><b>Concatenating -</b> We combine all phrases together by joining the phrases, creating a giant long continuous string.
      <ul>
        <li>
          <b>Expected Outcome:</b> <code>oceans are deep blue tomatoes are ripe red grass is green pennies are really shiny lakes are blue</code>
        </li>
      </ul>
    </li>
    <li><b>Converting the String into a Set of Characters -</b> Once the list of phrases is converted to a single long continuous string, we need to extract the unique characters from this string. 
      This is done by converting the string into a set, removing duplicates, and ensuring that each character in the string is represented once.
      <ul>
        <li>
          <b>Known Characters in Our Phrases & Expected Outcome:</b> <code>{'m', 'n', 'a', ' ', 'u', 'l', 'y', 's', 'h', 'c', 'r', 'k', 'o', 'p', 'i', 'g', 'e', 'd', 'b', 't'}</code>
        </li>
      </ul>
    </li>
    <li><b>Sorting -</b> Once we have a set of unique characters, we need to sort these characters in alphabetical order.
      <ul>
        <li>
          <b>Expected Outcome Sorted:</b> <code>[' ', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u', 'y']</code>
        </li>
      </ul>
    </li>
  </ul>

  </p>


Here is the Python code that demonstrates the implementation of the explanation provided above.

~~~python
unique_chars = sorted(set("".join(phrases))) 
~~~

### Create a Character-to-Index Mapping

Now that we a have unique list of characters, we need to create a character-to-index mapping. This is a dictionary that assigns a unique index to each character in the sorted list of characters.

~~~python
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)} # assigns an index to each character and assigns each unique character to its index.
~~~

~~~
Expected Outcome: {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'r': 15, 's': 16, 't': 17, 'u': 18, 'y': 19}
~~~


### Reverse Mapping

We reverse our character to index mapping, where each index points to its corresponding character.

~~~python
idx_to_char = {idx: char for char, idx in char_to_idx.items()} # Creates a reverse mapping where each index points its corresponding character.
~~~

~~~
Expected Outcome: {0: ' ', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'g', 7: 'h', 8: 'i', 9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'r', 16: 's', 17: 't', 18: 'u', 19: 'y'}
~~~


### Finding the Max Phrase Length

<p align="justify">
In order to process phrases uniformly, we must find the maximum length of any phrase in the dataset. This value will be used to pad shorter sequence, making sure they all have the same length. We're doing this to ensure our model is using fixed-length inputs. 
Having the max length ensures none of our data is truncated while regulating input dimensions.
</p>
  <ul>
    <li><b>"pennies are really shiny"</b> has the most characters compared to the other paraphrases in our dataset with a character count of 24.</li>
  </ul>


~~~python
max_len = max(len(phrase) for phrase in phrases)  # gets the max length of each phrase in the dataset
~~~

### Converting Phrases to Sequence of Indices

<p align="justify">
Each phrase in our list is then converted into a list of indicies/points, where each character within the phrase is replaced with its corresponding index from the character-to-index mapping. We do this because numerical representation is important 
when feeding text data into machine learning models. For example the phrase "oceans are deep blue" will be transformed to:
</p>

<ul>
  <li><code>[13, 3, 5, 1, 12, 16, 0, 1, 15, 5, 0, 4, 5, 5, 14, 0, 2, 10, 18, 5]</code></li>
  <li><code>o = 13, c = 3, e = 5, a = 1, n = 12, s = 16, SPACE = 0, a = 1, r = 15, e = 5, SPACE = 0, d = 4, e = 5, e = 5, p = 14, SPACE = 0, b = 2, l = 10, u = 18, e = 5</code></li>
</ul>

~~~python
sequences = [[char_to_idx[char] for char in phrase] for phrase in phrases]  # converts each phrase into a sequence into a sequence of indices based on character-to-index mapping. this turns each char in the phrase into its corresponding index.
~~~

### Padding Sequences to Uniform Length

<p align="justify">
To ensure phrases are the same length as the max length phrase, the sequences that are short are padded with zeros at the end to ensure all phrases havve the same length as the longest phrase.
<ul>
  <li>The phrase "pennies are really shiny" is the longest phrase with the most characters and has 24 characters at max.</li>
  <li>The phrase "oceans are deep blue", with only 20 characters, needs four 0s added at the end to reach the required length of 24 characters.</li>
</ul>
</p>

~~~python
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post') #pads the sequences to ensure same variable length. if a sequence is less than max length it be will be padded with zeros at the end.
~~~

### Preparing Our X and Y Variable Data

<ul>
  <li>X will consist all padded sequntia characters except the last one in each phrase.</li>
  <li> y will consist all padded sequntial characters except the first one in each phrase.
    <ul>
      <li>y data will go through one hot encoding (a method for converting categorical variables into a binary format).</li>
    </ul>
  </li>
</ul>

~~~python
X = padded_sequences[:, :-1] #inputs
y = padded_sequences[:, 1:] #outputs
y = np.array([to_categorical(seq, num_classes=len(unique_chars)) for seq in y]) # one hot enco
~~~

## Building Our RNN Model
<p align="justify">
Our simple RNN model is made up of layers upon layers. Our RNN model is made up of an Embedding Layer, Simple RNN Layer, and a Dense Layer.
<ul>
  <li> <b>Embedding Layer - </b>Purpose is to convert each input character into a dense vector representation of a fixed size of our output dimension (64 in this case).</li>
  <li> <b>Simple RNN Layer - </b>Processes the input sequence, step by step, while maintaining a hidden state that gathers information about the sequence.</li>
  <li><b>Dense Layer - </b> Acts as the output layer, providing a probability distribution over the vocabulary for each time step.</li>
</ul>
</p>

~~~python
model = Sequential([
    Embedding(input_dim=len(unique_chars), output_dim=64),
    SimpleRNN(128, return_sequences=True),
    Dense(len(unique_chars), activation='softmax')
])
~~~

## Compiling Our Model

~~~python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
~~~

## Training the RNN Model

We'll train our x and y data for 70 epochs, with a batchsize of 2, and verbose is set to true

~~~python
model.fit(X, y, epochs=70, batch_size=2, verbose=1)
~~~

After training, our RNN model should come out to be 98.75% accurate .

## Evaluating The RNN Model

<p align="justify">
We evaluated and visualized the model's capability of predicting and generating the next set of characters without any problems. The model effectively demonstrated its ability to handle sequential patterns by accurately predicting the next character in phrases.
</p>

**Predicting Printed Result**

~~~python
def generate_text(model, start_phrase, length):
    generated = start_phrase #start of the initial phrasing
    current_input = [char_to_idx[char] for char in start_phrase] #convert the starting phrase to a list of indicies based on the character-to-mapping

    for _ in range(length):
        current_input_padded = pad_sequences([current_input], maxlen=max_len - 1, padding='post') #pad all inputs to ensure it has the correct length for the model
        prediction = model.predict(current_input_padded, verbose=0) #Use the model to predict the next character
        next_char_idx = np.argmax(prediction[0, len(current_input) - 1]) #gets the next index of the predicted next character
        next_char = idx_to_char[next_char_idx] # map the predicted index back to the character
        generated += next_char #add the predited character to the generated text
        current_input.append(next_char_idx) #update the current input by adding the predicted character index
        if len(current_input) > max_len - 1: #make sure the input doesn't exceeds the model's input size
            current_input = current_input[1:] #remove the first character index to maintain the correct input length

    return generated[:length] #return the fully generated text


starting_word = "pen" #the first term we'll use to see what the model predicts based on this word.
generated_text = generate_text(model, starting_word, length=24) #use the model to predict the next set of character
print(f"Generated text using the starting word '{starting_word}': {generated_text}") #print our results
~~~

~~~
Output: Generated text using the starting word 'pen': pennies are really shiny
~~~

**Visualizing Predicted Result**

~~~python
def plot_generated_text(steps, chars):
    char_indices = [char_to_idx[char] for char in chars] #coverts characters to numerical points/indicies for plotting

    plt.figure(figsize=(12, 6)) #plot the results with a fill_between graph with a 12x6 plot.

    plt.fill_between(steps, char_indices, color="green", alpha=0.3, label="Generated Characters Flow") #fill between the generated character values and a baseline (e.g., zero or baseline value)


    plt.scatter(steps, char_indices, color="blue", label="Generated Characters (Scatter)", alpha=0.7) #scatter plot of the characters

    plt.plot(steps, char_indices, color="black", alpha=0.6, linewidth=1, label="Character Sequence") #connects scatter points with black lines.

    plt.axhline(0, color='black', linewidth=1, linestyle="--", label="Baseline") #represents a baseline


    plt.yticks(np.arange(len(char_to_idx)), [char for char, _ in sorted(char_to_idx.items())]) #set y-axis labels to characters

    plt.title("Text Generation Steps") #title of graph
    plt.xlabel("Step") #X axis label
    plt.ylabel("Character") #Y axis label
    plt.grid(alpha=0.3)


    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.) #move the legend to the far right

    plt.show()

#test usage
starting_word = "pen"  #the first term we'll use to see what the model predicts based on this word.
generated_text = generate_text(model, starting_word, length=24)  #use the model to predict the next set of characters

#prepare steps and chars for plotting
steps = list(range(len(starting_word))) + list(range(len(starting_word), len(starting_word) + len(generated_text) - len(starting_word)))
chars = list(starting_word) + list(generated_text[len(starting_word):])

plot_generated_text(steps, chars) #plot the generated text
~~~


<p align="center">
  <img src="https://github.com/dug22/Text-Generation-Using-an-RNN-Model/raw/main/images/text_generation_graph.png?raw=true">
</p>

It looks like our RNN model predicted the right text **pennies are really shiny** when our starting word was **pen**.

## Conclusion

<p align="justify">
Recurrent Neural Networks (RNNs) are well-suited for processing short-term dependencies in sequential data, making them effective for tasks where relationships between nearby elements are critical. However, they struggle significantly with long-term dependencies due to issues like vanishing gradients and limited memory capacity, which undermine their performance in capturing patterns over extended sequences. 
This limitation highlights the critical importance of advanced architectures like Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRUs), and Transformers.
</p>

## Source To Repository
[Text Generation Using Recurrent Neural Networks GitHub Repository](https://github.com/dug22/Text-Generation-Using-an-RNN-Model)

## References
[What is RNN?](https://aws.amazon.com/what-is/recurrent-neural-network/)

[Introduction to Recurrent Neural Network](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)

