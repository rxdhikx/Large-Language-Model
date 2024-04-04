import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read responses from a CSV file
def read_responses_from_csv(file_path):
    responses = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            responses.extend(row)
    return responses

# File paths for the CSV files
file_1_path = '/home/rravindra0463@id.sdsu.edu/models/100_res_trivia_web.csv'
file_2_path = '/home/rravindra0463@id.sdsu.edu/models/100_res_trivia_wiki.csv'

# Read responses from CSV files
dataset1_responses = read_responses_from_csv(file_1_path)
dataset2_responses = read_responses_from_csv(file_2_path)

# Concatenate responses from both datasets
all_responses = dataset1_responses + dataset2_responses

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(all_responses)

# Compute cosine similarity between datasets
cosine_sim = cosine_similarity(tfidf_matrix[:len(dataset1_responses)], tfidf_matrix[len(dataset1_responses):])

# Print cosine similarity matrix
print("Cosine Similarity Matrix:")
print(cosine_sim)
