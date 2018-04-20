import fuzzywuzzy
import re

# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio=90):
    # get a list of unique strings
    strings = df[column].unique()

    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match

    # let us know the function's done
    print("All done!")

# This function return a darray of unique values of a dataframe column
# it takes a dataframe and column name
def get_column_unique_entry_size(df, column):
    entries = df[column].unique()
    entries.sort()
    print(entries.size)
    return entries

# This function takes a list of string (date with expected format 08/08/1888), it checks if all elements repects this format
# or not, it takes a list of string (date), print the wrong format string, and the correct format string count.
def getWrongDateFormat(dateList):
    n=0
    for date in dateList:
        pattern = re.compile("[0-9]+/[0-9]+/[0-9]+")
        if pattern.match(date):
            n=n+1
        else:
            print(date)
    print("Correct format date size : "+str(n))
    print("All date size : "+str(len(dateList)))
