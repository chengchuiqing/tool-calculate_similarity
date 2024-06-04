# coding:utf-8

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def dict_demo():
    """
    字典特征提取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 字典特征提取
    # 1.实例化
    transfer = DictVectorizer(sparse=False)

    # 2.调用fit_transform
    trans_data = transfer.fit_transform(data)

    print("特征名字是：\n", transfer.get_feature_names_out())

    print(trans_data)


def english_count_text_demo():
    """
    文本特征提取 -- 英文
    :return: NOne
    """
    data = ["life is is short,i like python",
            "life is too long,i dislike python"]

    # 1.实例化
    # transfer = CountVectorizer(sparse=False)  # 注意，没有sparse
    transfer = CountVectorizer(stop_words=["dislike"])

    # 2.调用fit_transform
    transfer_data = transfer.fit_transform(data)

    print(transfer.get_feature_names_out())
    print(transfer_data.toarray())


def chinese_count_text_demo1():
    """
    文本特征提取 -- 中文
    :return: NOne
    """
    data = ["人生 苦短, 我 喜欢 Python",
            "生活 太长久, 我 不 喜欢 Python"]

    # 1.实例化
    transfer = CountVectorizer()

    # 2.调用fit_transform
    transfer_data = transfer.fit_transform(data)

    print(transfer.get_feature_names_out())
    print(transfer_data.toarray())


def cut_word(sen):
    """
    中文分词
    :return: sen
    """
    # print(" ".join(list(jieba.cut(sen))))
    return " ".join(list(jieba.cut(sen)))


def chinese_count_text_demo2():
    """
    文本特征提取 -- 中文
    :return: NOne
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    list = []
    for temp in data:
        # print(temp)
        list.append(cut_word(temp))
    print(list)

    # 1.实例化
    transfer = CountVectorizer(stop_words=["一种", "还是"])

    # 2.调用fit_transform
    transfer_data = transfer.fit_transform(list)

    print(transfer.get_feature_names_out())
    print(transfer_data.toarray())


def tfidf_text_demo():
    """
    文本特征提取 -- 中文
    :return: NOne
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    list = []
    for temp in data:
        # print(temp)
        list.append(cut_word(temp))
    # print(list)

    # 1.实例化
    # transfer = CountVectorizer(stop_words=["一种", "还是"])
    transfer = TfidfVectorizer()

    # 2.调用fit_transform
    transfer_data = transfer.fit_transform(list)

    print(transfer.get_feature_names_out())
    print(transfer_data.toarray())


if __name__ == '__main__':
    # dict_demo()
    # english_count_text_demo()
    # chinese_count_text_demo1()
    # print(cut_word("我喜欢你中国"))
    # chinese_count_text_demo2()
    tfidf_text_demo()
