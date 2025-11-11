import pandas as pd
import re
from collections import Counter

from nltk import word_tokenize
# אם אין לך את הספרייה wordcloud תצטרכי להתקין אותה עם pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("../CSVfiles/balanced_mbti.csv")
ENTJ_ESTJ_df = df[(df['type'] == 'ENTJ') | (df['type'] == 'ESTJ')]
non_assertive_df = df[(df['type'] != 'ENTJ') & (df['type'] != 'ESTJ')]

# combine all posts from both groups
assertive_text = ' '.join(ENTJ_ESTJ_df['posts'])
non_assertive_text = ' '.join(non_assertive_df['posts'])

# split into words
assertive_words = assertive_text.split()
non_assertive_words = non_assertive_text.split()

# words counting
assertive_word_counts = Counter(assertive_words)
non_assertive_word_counts = Counter(non_assertive_words)

# sorting assertive words by frequency (most common first)
sorted_assertive = sorted(assertive_word_counts.items(), key=lambda x: x[1], reverse=True)

# printing each word: number of occurrences in assertive and non-assertive
with open("output.txt", "w", encoding="utf-8") as f:
    for word, assertive_count in sorted_assertive:
        non_assertive_count = non_assertive_word_counts.get(word, 0)
        total = assertive_count + non_assertive_count
        if total == 0:
            continue
        assertive_pct = assertive_count / total * 100
        if 60 < assertive_pct < 100 and assertive_count > 50:
            non_assertive_pct = non_assertive_count / total * 100
            line = (f"{word}: {assertive_count}/{total} ({assertive_pct:.2f}%) "
                    f"(non-assertive: {non_assertive_count}/{total} ({non_assertive_pct:.2f}%))\n")
            f.write(line)

# # printing the 70 most common words
# for word, count in word_counts.most_common(70):
#     print(f'{word}: {count}')
#
# # create graphical word cloud:
# wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(word_counts.most_common(70)))
#
# plt.figure(figsize=(12, 6))
# plt.imshow(wc, interpolation='bilinear')
# plt.axis('off')
# plt.show()

assertive_words = [
    # שליטה, הישג, סמכות
    "power", "leadership", "authority", "decision", "success",
    "confidence", "achieve", "goal", "challenge",

    # פעולה או יוזמה
    "improve", "fix", "manage", "accomplish", "pull",
    "sell", "leader","leadership", "offer", "insert",

    # עימות, קונפליקט, עמדה
    "debate", "battle", "rebel", "enemy", "risk", "charge",
    "aggressive", "intimidate", "challenge", "strategy",

    # שליטה עצמית / התקדמות
    "mature", "prioritize", "ambition", "productive",
    "effective", "manage", "master", "professional", "improvement",

    # ערך סמלי/חברתי גבוה
    "respect", "responsibility", "selfesteem", "ambition", "award",
    "credit", "confidence", "vision", "authority", "powerful"
]