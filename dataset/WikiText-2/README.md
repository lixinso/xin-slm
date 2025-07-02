# WikiText-2 Dataset

The WikiText-2 dataset is a popular benchmark for language modeling, sourced from high-quality, verified "Good" and "Featured" articles on Wikipedia. The text is preserved in a mostly raw format, which includes structural elements like section headers (e.g., ` = = Section Name = = `), making it a valuable resource for training models that can understand semi-structured text.

The dataset is divided into three splits:

*   **`train`**: The largest portion of the data, used for training the language model.
*   **`validation`**: A smaller, separate set used to tune the model's hyperparameters and prevent overfitting.
*   **`test`**: A final, held-out set used to evaluate the model's performance after training is complete.

## Examples

Here are some examples to illustrate the structure and content of the data in each split:

---

### **Train Split Example:**

This is a typical entry from the training set. Notice the title formatting.

```
 = Valkyria Chronicles = 

 Valkyria Chronicles is a tactical role @-@ playing game developed and published by Sega , exclusively for the PlayStation 3 . The game was released in Japan on April 24 , 2008 , in North America on November 4 , 2008 , and in Europe and Australia on October 31 , 2008 . A Microsoft Windows version was released on November 11 , 2014 . A remastered version for PlayStation 4 was released in Japan on February 10 , 2016 and later in North America and Europe in May 2016 .
```

---

### **Validation Split Example:**

This is an example from the validation set, used to tune the model during training.

```
 = = Plot = = 

 The game is set in a fictional 1935 , on a continent reminiscent of Europe called Europa . The neutral nation of Gallia , a constitutional monarchy , is caught in a war between the East Europan Imperial Alliance and the Atlantic Federation over the precious mineral Ragnite . The Imperial army , led by Prince Maximillian , successfully invades Gallia and is in the process of taking over the country . Welkin Gunther , son of the late General Belgen Gunther , is forced to fight for his life alongside Alicia Melchiott , captain of the Bruhl town watch .
```

---

### **Test Split Example:**

This is an example from the test set, used for the final evaluation of the model's performance.

```
 = = Reception = = 

 Valkyria Chronicles received generally positive reviews from critics , with an average of 86 / 100 on Metacritic . The game was praised for its innovative gameplay , cel @-@ shaded art style , and story . It was awarded " Best RPG of 2008 " by GameSpot and " Strategy Game of the Year " by GameSpy . However , the game was a commercial failure in North America , selling only 33,000 copies in its first month .
```
