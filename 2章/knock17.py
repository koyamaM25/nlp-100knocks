file_name = 'popular-names.txt'
unique_names = set() 
output_file = 'out_17.txt'

with open(file_name, 'r', encoding='utf-8') as f:
    for line in f:
        columns = line.split('\t')
        
        if columns and columns[0]:
            unique_names.add(columns[0])
    
print(f"--- ファイル '{file_name}' の1列目の異なり（文字列の種類） ---")
    
with open(output_file, 'w', encoding = 'utf-8') as out_f:
    out_f.writelines(unique_names)
    
print(f"\nユニークな文字列の総数: {len(unique_names)}")

# $ cut -f 1 popular-names.txt | sort | uniq
# Abigail
# Aiden
# Alexander
# Alexis
# Alice
# Amanda
# Amelia
# Amy
# Andrew
# Angela
# Anna
# Annie
# Anthony
# Ashley
# Austin
# Ava
# Barbara
# Benjamin
# Bertha
# Bessie
# Betty
# Brandon
# Brian
# Brittany
# Carol
# Carolyn
# Charles
# Charlotte
# Chloe
# Christopher
# Clara
# Crystal
# Cynthia
# Daniel
# David
# Deborah
# Debra
# Donald
# Donna
# Doris
# Dorothy
# Edward
# Elijah
# Elizabeth
# Emily
# Emma
# Ethan
# Ethel
# Evelyn
# Florence
# Frances
# Frank
# Gary
# George
# Hannah
# Harper
# Harry
# Heather
# Helen
# Henry
# Ida
# Isabella
# Jacob
# James
# Jason
# Jayden
# Jeffrey
# Jennifer
# Jessica
# Joan
# John
# Joseph
# Joshua
# Judith
# Julie
# Justin
# Karen
# Kathleen
# Kelly
# Kimberly
# Larry
# Laura
# Lauren
# Liam
# Lillian
# Linda
# Lisa
# Logan
# Lori
# Lucas
# Madison
# Margaret
# Marie
# Mark
# Mary
# Mason
# Matthew
# Megan
# Melissa
# Mia
# Michael
# Michelle
# Mildred
# Minnie
# Nancy
# Nicholas
# Nicole
# Noah
# Oliver
# Olivia
# Pamela
# Patricia
# Rachel
# Rebecca
# Richard
# Robert
# Ronald
# Ruth
# Samantha
# Sandra
# Sarah
# Scott
# Sharon
# Shirley
# Sophia
# Stephanie
# Steven
# Susan
# Tammy
# Taylor
# Thomas
# Tracy
# Tyler
# Virginia
# Walter
# William