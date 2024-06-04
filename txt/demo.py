import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def write_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


def split_into_paragraphs(text):
    paragraphs = text.split('\n\n')
    return [p for p in paragraphs if p.strip() != '']


def calculate_similarity(paragraphs1, paragraphs2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(paragraphs1 + paragraphs2)

    similarity_matrix = cosine_similarity(tfidf_matrix[:len(paragraphs1)], tfidf_matrix[len(paragraphs1):])
    return similarity_matrix


def main():

    # 读取文件内容
    content_a = read_file('A.txt')  # 原文本
    content_b = read_file('B.txt')  # 待查重文本

    # 分段
    paragraphs_a = split_into_paragraphs(content_a)
    paragraphs_b = split_into_paragraphs(content_b)

    # 计算相似性
    similarity_matrix = calculate_similarity(paragraphs_a, paragraphs_b)

    # 输出相似性矩阵
    for i, row in enumerate(similarity_matrix):
        for j, similarity in enumerate(row):
            print(f"原文第{i + 1}段落 和 待查重文本第{j + 1}段落 的相似度: {similarity:.4f}")

    # 输出相似的内容
    similar_content = []
    threshold = 0.5  # 相似度阈值
    for i, row in enumerate(similarity_matrix):
        for j, similarity in enumerate(row):
            if similarity > threshold:
                similar_content.append(
                    f"原文第{i + 1}段落:\n{paragraphs_a[i]}\n\n和\n\n待查重文本第{j + 1}段落:\n{paragraphs_b[j]}\n\n相似度: {similarity:.4f}\n\n{'=' * 50}\n\n")

    write_file('result.txt', ''.join(similar_content))


if __name__ == "__main__":
    main()
