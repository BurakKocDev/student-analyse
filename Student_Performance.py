import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("StudentsPerformance.csv")

# Initial Data Exploration
print(data.head(5))
print(data.shape)
print(data.describe())
print(data.isnull().sum())
print(data.columns)

# MATH
plt.figure()  # Yeni bir grafik alanı oluşturur
p = sns.countplot(x="math score", data=data, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()
plt.clf()  # Grafiği kapatır

passmark = 50
data["Math_PassStatus"] = np.where(data["math score"] < passmark, "failed", "successful")
print(data.Math_PassStatus.value_counts())

plt.figure()
p = sns.countplot(x="parental level of education", data=data, hue="Math_PassStatus", palette="bright")
p.set_xticklabels(p.get_xticklabels(), rotation=45, ha="right")
plt.show()
plt.clf()

# READING
plt.figure()
p = sns.countplot(x="reading score", data=data, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()
plt.clf()

data["Reading_PassStatus"] = np.where(data["reading score"] < passmark, "failed", "successful")
print(data.Reading_PassStatus.value_counts())

plt.figure()
p = sns.countplot(x="parental level of education", data=data, hue="Reading_PassStatus", palette="bright")
p.set_xticklabels(p.get_xticklabels(), rotation=45, ha="right")
plt.show()
plt.clf()

# WRITING
plt.figure()
p = sns.countplot(x="writing score", data=data, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()
plt.clf()

data["Writing_PassStatus"] = np.where(data["writing score"] < passmark, "failed", "successful")
print(data.Writing_PassStatus.value_counts())

plt.figure()
p = sns.countplot(x="parental level of education", data=data, hue="Writing_PassStatus", palette="bright")
p.set_xticklabels(p.get_xticklabels(), rotation=45, ha="right")
plt.show()
plt.clf()

# Overall Pass Status
data["Overall_PassStatus"] = data.apply(lambda x: "failed" if x["Math_PassStatus"] == "failed" or x["Reading_PassStatus"] == "failed" or x["Writing_PassStatus"] == "failed" else "successful", axis=1)
print(data.Overall_PassStatus.value_counts())

plt.figure()
p = sns.countplot(x="parental level of education", data=data, hue="Overall_PassStatus", palette="bright")
p.set_xticklabels(p.get_xticklabels(), rotation=45, ha="right")
plt.show()
plt.clf()

# Total Marks and Percentage
data["Total_Makrs"] = data["math score"] + data["reading score"] + data["writing score"]
data["Percenrage"] = data["Total_Makrs"] / 3
print(data.columns)

plt.figure()
p = sns.countplot(x="Percenrage", data=data, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.show()
plt.clf()

# Grade Calculation
def GetGrade(Percenrage, Overall_PassStatus):
    if Overall_PassStatus == "failed":
        return "failed"
    if Percenrage >= 80:
        return "A"
    elif Percenrage >= 70:
        return "B"
    elif Percenrage >= 60:
        return "C"
    elif Percenrage >= 50:
        return "D"
    elif Percenrage >= 40:
        return "E"
    else:
        return "F"

data["Grade"] = data.apply(lambda x: GetGrade(x["Percenrage"], x["Overall_PassStatus"]), axis=1)
print(data.Grade.value_counts())

plt.figure()
sns.countplot(x="Grade", data=data, order=["A", "B", "C", "D", "E", "F"], palette="muted")
plt.show()
plt.clf()

plt.figure()
p = sns.countplot(x="parental level of education", data=data, hue="Grade", palette="bright")
p.set_xticklabels(p.get_xticklabels(), rotation=45, ha="right")
plt.show()
plt.clf()


print(data)

data_gender =data["gender"]
female_count =data_gender.value_counts()["female"]
male_count =data_gender.value_counts()["male"]


female_ratio = female_count/(female_count/male_count)*100
male_ratio = male_count/(male_count/female_count)*100

print("proportion of girls :",female_ratio)
print("proportion of boys :", male_ratio )

labels = ["girl","boy"]
sizes = [female_ratio,male_ratio]
colors = ["pink","lightblue"]
plt.pie(sizes,labels = labels,colors = colors,autopct="%1.1f%%",startangle=90)
plt.axis("equal")
plt.title("girl-boy ratios")
plt.show()

#girls and boys math success rate
 
data_female = data[data["gender"] == "female"]
data_male = data[data["gender"] == "male"]

female_math = data_female["math score"].mean()
male_math = data_male["math score"].mean()

# Correcting variable names here
female_to_male_math_ratio = female_math / male_math
print("Ratio of girls to boys in math:", female_to_male_math_ratio)

import matplotlib.pyplot as plt

# We use the female_math and male_math variables defined earlier
labels = ["girls", "boys"]
values = [female_math, male_math]
colors = ["pink", "lightblue"]

plt.bar(labels, values, color=colors)

plt.axhline(y=male_math, color="gray", linestyle="--")
plt.text(-0.1, male_math, "average (boys)", ha="right")

plt.axhline(y=female_math, color="gray", linestyle="--")
plt.text(-0.1, female_math, "average (girls)", ha="right")
plt.text(-0.5, male_math * 1.05, "ratio: {:.2f}".format(female_to_male_math_ratio), ha="center")

plt.xlabel("Gender")
plt.ylabel("Math Grades")
plt.title("The ratio of girls' success in math to boys' success")
plt.show()


import matplotlib.pyplot as plt

parental_edu_counts = data["parental level of education"].value_counts()
parental_edu_levels = parental_edu_counts.index.tolist()

# Collecting math scores
math_scores = []
for level in parental_edu_levels:
    math_scores.append(data[data["parental level of education"] == level]["math score"].mean())

# We draw the bar graph
plt.bar(parental_edu_levels, math_scores)
plt.xlabel("Family Education Level")
plt.ylabel("math average")
plt.title("Family Education and Mathematics")
plt.xticks(rotation=90)  
plt.show()


#reading score

import matplotlib.pyplot as plt

parental_edu_counts = data["parental level of education"].value_counts()
parental_edu_levels = parental_edu_counts.index.tolist()

# Collecting reading scores
reading_scores= []
for level in parental_edu_levels:
    reading_scores.append(data[data["parental level of education"] == level]["reading score"].mean())

# We draw the bar graph
plt.bar(parental_edu_levels, reading_scores)
plt.xlabel("Family Education Level")
plt.ylabel("reading average")
plt.title("Family Education and reading")
plt.xticks(rotation=90)  
plt.show()


#writing score

import matplotlib.pyplot as plt

parental_edu_counts = data["parental level of education"].value_counts()
parental_edu_levels = parental_edu_counts.index.tolist()

# Collecting writing scores
writing_scores = []
for level in parental_edu_levels:
    writing_scores.append(data[data["parental level of education"] == level]["writing score"].mean())

# We draw the bar graph
plt.bar(parental_edu_levels, math_scores)
plt.xlabel("Family Education Level")
plt.ylabel("writing average")
plt.title("Family Education and writing")
plt.xticks(rotation=90)  
plt.show()





#the influence of ethnicity on mat
print(data.columns)

ethnicity  = data.groupby("race/ethnicity")["math score"].mean().reset_index()
plt.bar(ethnicity ["race/ethnicity"],ethnicity ["math score"])
plt.title("the influence of ethnicity on mat")
plt.xlabel("ethnicity ")
plt.ylabel("math")

plt.show()


#mat grades by ethnicity and family level

ethnicity  = data.groupby(["race/ethnicity","parental level of education"])["math score"].mean().reset_index()
sns.catplot(x="parental level of education",y = "math score",hue = "race/ethnicity",kind = "bar",data = ethnicity,height = 6,aspect =1.5)
plt.title("the influence of ethnicity on mat")
plt.xlabel("ethnicity ")
plt.ylabel("math")

plt.show()

