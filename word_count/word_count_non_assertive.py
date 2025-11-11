import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# load the CSV file
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

# sorting non-assertive words by frequency (most common first)
sorted_non_assertive = sorted(non_assertive_word_counts.items(), key=lambda x: x[1], reverse=True)

# printing each word: number of occurrences in non-assertive and assertive
with open("output.txt", "w", encoding="utf-8") as f:
    for word, non_assertive_count in sorted_non_assertive:
        assertive_count = assertive_word_counts.get(word, 0)
        total = non_assertive_count + assertive_count
        if total == 0:
            continue
        non_assertive_pct = non_assertive_count / total * 100
        if 60 < non_assertive_pct < 100 < non_assertive_count:
            assertive_pct = assertive_count / total * 100
            line = (f"{word}: {non_assertive_count}/{total} ({non_assertive_pct:.2f}%) "
                    f"(assertive: {assertive_count}/{total} ({assertive_pct:.2f}%))\n")
            f.write(line)

# # printing the 70 most common words
# for word, count in word_counts.most_common(70):
#     print(f'{word}: {count}')
#

non_assertive_words = [
    # רגשות ומצבים פנימיים
    "sad", "dream", "alone", "anxious", "nervous", "panic", "overwhelm",
     "mood", "sensation", "beauty", "nightmare", "impulsive",
    "breathe", "awkward", "uncomfortable", "embarrass", "messy", "introversion",

    # תרבות ובידור
    "music", "song", "guitar", "band", "lyric", "album", "tv", "anime",
    "comedy", "artist", "genre", "piano", "instrument",

    #  שפת אינטרנט / ביטויים לא פורמליים
    "lol", "haha", "lmao", "hehe", "heh", "wtf", "idk", "omg",
    "dunno", "yay", "yeah", "yea", "btw",

    # ️ יחסים בין-אישיים
    "brother", "wife", "hug", "roommate", "hello", "hi", "smile",
    "quiet", "awkward",

    #  מקומות / לאום / פוליטיקה
    "german", "germany", "europe", "uk", "eu", "france", "russian",
    "russia", "town", "local", "city", "church", "election",
    "socialist", "western", "foreign",

    #  זהות / מחשבות פילוסופיות
    "consciousness", "soul", "deeply", "imagination", "wonderful",
    "beauty", "harmony", "ramble", "mainly",

    #  גוף וחושים
    "finger", "ear", "smell", "sensation", "breathe",

    #  הרגלים ותחביבים
    "smoke", "weed", "alcohol", "medication", "bike", "park",
    "sing", "texting", "photo", "tv",

    #  תחושות כלליות או מצבים
    "weird", "quiet", "sad", "cute", "awkward",
    "warm", "alone", "beautiful", "messy", "uncomfortable",

    # טכנולוגיה ותקשורת
    "phone", "texting", "click", "loop", "delete",

    #  מילים כלליות אך אינדיקטיביות
    "kinda", "anymore", "lately", "nowadays", "alot", "wtf",
    "idk", "dunno", "btw"
]
print(len(non_assertive_words))