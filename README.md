# Reverse-Image-Search

This program uses the Flux package to load a pre-trained CNN model and extract features from an image, and the SQLite package to connect to a SQLite database and perform a search for similar images based on the extracted features. The ImageMagick package is used to read in an image file from the filesystem.

To use this program, you will need to have a database with a table of images and their corresponding features, as well as the pre-trained CNN model. You can then call the **reverse_image_search** function with the features of the query image and the path to the database, and it will return a list of the most similar images based on their cosine similarity to the query image.

