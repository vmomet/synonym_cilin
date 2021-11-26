'''
Description:一种基于层次结构的同义词林同义词、同类词查询方法
Author: vmomet
Date: 2021-11-26 11:18
'''
import random

random.seed(42)


class SynonymCilin(object):
    """ 基于哈工大同义词词林扩展版计算语义相似度，结合字典树方便同义词、同类词的查询 """

    def __init__(self, cilin_path):
        """
        'code_word' 以编码为key，单词list为value的dict，一个编码有多个单词
        'word_code' 以单词为key，编码为value的dict，一个单词可能有多个编码
        'vocab' 所有不重复的单词，便于统计词汇总数。
        'N' 为单词出现累计次数，含重复出现的。
        'trie' 词林字典树，便于相似词的查找
        """
        self.N = 0
        self.code_word = {}
        self.word_code = {}
        self.vocab = set()
        self.read_cilin(cilin_path)
        self.trie = CilinTrie()
        self.create_cilin_trie()

    def read_cilin(self, cilin_path):
        """
        读入同义词词林，编码为key，词群为value，保存在self.code_word
        单词为key，编码群为value，保存在self.word_code
        所有单词保存在self.vocab
        """
        with open(cilin_path, 'r', encoding='gbk') as f:
            for line in f.readlines():
                res = line.split()
                code = res[0]  # 词义编码
                words = res[1:]  # 同组的多个词
                self.vocab.update(words)  # 一组词更新到词汇表中
                self.code_word[code] = words  # 字典，目前键是词义编码，值是一组单词。

                self.N += len(words)  # 包含多义词的总义项数。
                for w in words:
                    if w in self.word_code.keys():  # 最终目的：键是单词本身，值是词义编码。
                        self.word_code[w].append(code)  # 如果单词已经在，就把当前编码增加到字典中
                    else:
                        self.word_code[w] = [code]  # 反之，则在字典中添加该项。

    def get_common_str(self, c1, c2):
        """        获取两个字符的公共部分，注意有些层是2位数字        """
        res = ''
        for i, j in zip(c1, c2):
            if i == j:
                res += i
            else:
                break
        if 3 == len(res) or 6 == len(res):
            res = res[:-1]
        return res

    def get_layer(self, common_str):
        """
        根据common_str返回两个编码【公共父节点的孩子】所在的层好。从下往上为0--5。
        如果没有共同的str，则位于第5层；
        如果第1个字符相同，则位于第4层；
        注意：最底层用0表示。
        """
        length = len(common_str)
        table = {1: 4, 2: 3, 4: 2, 5: 1, 7: 0}
        if length in table.keys():
            return table[length]
        return 5

    def code_layer(self, c):
        """将编码按层次结构化：    Aa01A01=
        第三层和第五层是两个数字表示；
        第一、二、四层分别是一个字母；
        最后一个字符用来区分所有字符相同的情况。
        """
        return [c[0], c[1], c[2:4], c[4], c[5:7], c[7]]

    def get_k(self, c1, c2):
        """        返回两个编码对应分支的距离，相邻距离为1        """
        if c1[0] != c2[0]:
            return abs(ord(c1[0]) - ord(c2[0]))
        elif c1[1] != c2[1]:
            return abs(ord(c1[1]) - ord(c2[1]))
        elif c1[2] != c2[2]:
            return abs(int(c1[2]) - int(c2[2]))
        elif c1[3] != c2[3]:
            return abs(ord(c1[3]) - ord(c2[3]))
        else:
            return abs(int(c1[4]) - int(c2[4]))

    def get_n(self, common_str):
        """
        计算所在分支层的分支数; 即计算分支的父节点总共有多少个子节点
        两个编码的common_str决定了它们共同处于哪一层
        例如，它们的common_str为前两层，则它们共同处于第3层，则统计前两层为common_str的第三层编码个数就好了
        """
        if not common_str:
            return 14  # 如果没有公共子串，则第5层有14个大类。

        siblings = set()
        layer = self.get_layer(common_str)  # 找到公共父节点的孩子所在层号
        for c in self.code_word.keys():
            if c.startswith(common_str):  # 如果遇到一个编码是以公共子串开头的
                clayer = self.code_layer(c)  # 找到该编码的一组编码。
                siblings.add(clayer[5 - layer])  # 将该公共子串后面的一个编码添加到集合当中。
        return len(siblings)

    def sim2016(self, w1, w2):
        """    按照论文《基于路径与深度的同义词词林词语相似度计算》_陈宏朝、李飞、朱新华等 2016年9月  """
        # 如果有一个词不在词林中，则相似度为0
        if w1 not in self.vocab or w2 not in self.vocab:
            return 0
        # 获取两个词的编码
        code1 = self.word_code[w1]
        code2 = self.word_code[w2]
        sim_max = 0
        for c1 in code1:  # 选取相似度最大值
            for c2 in code2:
                cur_sim = self.sim2016_by_code(c1, c2)
                sim_max = cur_sim if cur_sim > sim_max else sim_max
        return sim_max

    def sim2016_by_code(self, c1, c2):
        """        根据编码计算相似度        """
        # 先把code的转换为一组类别编号。
        clayer1, clayer2 = self.code_layer(c1), self.code_layer(c2)
        # 找到公共字符串
        common_str = self.get_common_str(c1, c2)
        layer = self.get_layer(common_str) + 1  # 找到公共父节点所在层号

        Weights = [0, 0.5, 1.5, 4, 6, 8]  # 在0号位置补0，方便层序号与下标一一对应。

        if len(common_str) >= 7:  # 如果前7个字符相同，则第8个字符也相同，要么同为'='，要么同为'#''
            if common_str[-1] == '=':
                return 1
            elif common_str[-1] == '#':
                return 0.5

        k = self.get_k(clayer1, clayer2)
        n = self.get_n(common_str)
        Depth = 0.9 if layer > 5 else sum(Weights[layer:])  # 从layer层，累加到最后
        Path = 2 * sum(Weights[:layer])  # 要从1，累加到layer-1这一层

        # print(common_str, 'K=', k, 'N=', n, '公共父节点层号：', layer, end = '\t')
        beta = k / n * Weights[layer - 1]  # 公共父节点所在层的权重
        return (Depth + 0.9) / (Depth + 0.9 + Path + beta) if layer <= 5 else 0.9 / (0.9 + Path + beta)

    def create_cilin_trie(self):
        """ 构造词林字典树 """
        for code in self.code_word.keys():
            self.trie.insert(code)
        # print("Num of code: ", len(self.code_word.keys()))
        # print("Create Cilin Trie done!")

    def get_synonym_words(self, word):
        """
        获取同义词
            根据词林的定义，最后一层为"="的词语属于同义词；
            为"#"属于同类词；为"@"属于反义词，且它们单独
            成词，没有词结伴。
        Example:
            >>> res = sc.get_synonym_words('苹果')
        """
        if word not in self.vocab:
            return []
        synonym_words = set()  # 同义词
        for code in self.word_code[word]:
            if code[-1] == "=":
                synonym_words.update(
                    [w for w in self.code_word[code] if w != word])
        return list(synonym_words)

    def get_similar_words(self, word, n=4, min_sim=0.5):
        """
        获取相似词
        不同于同义词，在这里近义词指同类但不同义

        :param word: 查询词
        :param n: 通过词林的层数选择近义词
                  可以将min_sim设置的比较小，以达到仅通过层数n来筛选近义词的目的
        :param min_sim: 通过相似度选择近义词
                        可以将n设置较小，以达到仅通过设置最小相似度min_sim来筛选近义词的目的
                        n不推荐设置为1或者2，这样搜索会比较慢
        :return: 近义词表

        Example:
            >>> res_by_sim = sc.get_similar_words('苹果', min_sim=0.9)
            >>> res_by_n = sc.get_similar_words('野生', n=6)
        """
        if word not in self.vocab:
            return []
        assert 1 <= n <= 6
        similar_words = set()
        codes = self.word_code[word]

        if n == 6:  # n==6意味着寻找带 "#"标签的同类词
            codes = [c for c in codes if c[-1] == "#"]
        for code in codes:
            code_layer = self.code_layer(code)
            prefix = ''.join(code_layer[:n])  # limited by n
            res = self.trie.get_start(prefix)  # 返回符合条件的所有code
            for code2 in res:
                sim = self.sim2016_by_code(code, code2)
                if sim >= min_sim and (n == 6 or word not in self.code_word[code2]):
                    similar_words.update(self.code_word[code2])
        if word in similar_words:
            similar_words.remove(word)
        return list(similar_words)

    def get_one_similar_word(self, word):
        """ 随机获取一个相似词 """
        if word not in self.vocab:
            return None

        # 优先从#标签获取
        codes = [c for c in self.word_code[word] if c[-1] == "#"]
        if len(codes) == 0:
            codes = self.word_code[word]
        for n in [6, 5, 4]:
            for code in codes:
                code_layer = self.code_layer(code)
                prefix = ''.join(code_layer[:n])  # limited by n
                res = self.trie.get_start(prefix)  # 返回符合条件的所有code
                if len(res) > 0:
                    rand_code = random.choice(res)
                    return random.choice(self.code_word[rand_code])
        return None


class TrieNode:
    def __init__(self):
        self.data = {}
        self.is_word = False


class CilinTrie:
    """ 用于构造词林的字典树 """

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def code_layer(self, c):
        """将编码按层次结构化：    Aa01A01=
        第三层和第五层是两个数字表示；
        第一、二、四层分别是一个字母；
        最后一个字符用来区分所有字符相同的情况。
        """
        ret = [None] * 6
        if len(c) >= 1:
            ret[0] = c[0]
        if len(c) >= 2:
            ret[1] = c[1]
        if len(c) >= 4:
            ret[2] = c[2:4]
        if len(c) >= 5:
            ret[3] = c[4]
        if len(c) >= 7:
            ret[4] = c[5:7]
        if len(c) >= 8:
            ret[5] = c[7]
        return ret

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        assert len(word) == len("Aa01A01=")
        node = self.root
        for chars in self.code_layer(word):
            child = node.data.get(chars)
            if not child:
                node.data[chars] = TrieNode()
            node = node.data[chars]
        node.is_word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        prefix_layers = [layer for layer in self.code_layer(word) if layer is not None]
        for chars in prefix_layers:
            node = node.data.get(chars)
            if not node:
                return False
        return node.is_word  # 判断单词是否是完整的存在在trie树中

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        prefix_layers = [layer for layer in self.code_layer(prefix) if layer is not None]
        for chars in prefix_layers:
            node = node.data.get(chars)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        """
          Returns words started with prefix
          :param prefix:
          :return: words (list)
        """

        def get_key(pre, pre_node):
            word_list = []
            if pre_node.is_word:
                word_list.append(pre)
            for x in pre_node.data.keys():
                word_list.extend(get_key(pre + str(x), pre_node.data.get(x)))
            return word_list

        words = []
        if not self.startsWith(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        prefix_layers = [layer for layer in self.code_layer(prefix) if layer is not None]
        for chars in prefix_layers:
            node = node.data.get(chars)
        return get_key(prefix, node)


if __name__ == '__main__':
    sc = SynonymCilin("data/cilin.txt")

    # 同义词查询
    res = sc.get_synonym_words('苹果')
    print(res)

    # 同类词（相似词）查询
    res = sc.get_similar_words('苹果', min_sim=0.9)
    print(res)
    res = sc.get_similar_words('野生', n=6)
    print(res)
