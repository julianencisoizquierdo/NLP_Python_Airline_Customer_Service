# NLP_Python_Airline_Customer_Service

For Case 1, we are tasked with analyzing the tweets of the competitor in the airline industry. We are provided with the airlines.csv and are asked a series of questions regarding the interactions.

To start off, we did some Exploratory Data Analysis to find out more about the information contained in this file. We realize that:
The dataset has 1,377 rows.
There are no repeated instances.
The interactions took place between the 1st and the 15th of October 2015.
The text column starts with an @ followed by a username. This must represent the name of the person who wrote a tweet about one of the competitor airlines.
The text column ends with two capital letters, which most likely represent the initials of the name and surname of the customer service agent that replied to the tweet.
Sometimes, the text row contains a fraction (eg: 2/2, 3/4, 2/4…) just after the initials of the agent’s initials. After grouping by username and replier, we realize that it means that it has to be combined with other rows in the text column in order to form a full tweet. For example, if a fraction 3/4 is found, it is telling us that this text row has to be combined with three other rows of the text column (two of which precede it, one of which comes after it) in order to see the full tweet.


Question 1 - What is the average length of a social customer service reply?

To answer this question, our initial idea was to join the messages that have been split, and only after that clean the text and compute the average length. However, this is not only difficult to do (given that we would have to group by username and agent and then try to match the meanings), but also impossible to do for all of the cases. As we will see, there are a few messages that are incomplete.

In order to get a better view of the messages that have been split, we create a column with the username and another one called count_username. This tells us how many times that username is present in the dataset. We first extract the username with str.extract(r'@(\w+)'), then create another dataframe where we group by username to obtain the counts, and we finally map this to the initial dataset.

Then, we extract the fractions that indicate if the text is part of another string or not. We call this new column iteration_id. For that, we use the regex expression (?<= )(\d/\d). The second part of the expression (\d/\d) looks for a digit (\d) that is followed by a slash, that is then followed by another digit. It is fundamental to add to this the lookbehind assertion (?<= ). Not doing so would sometimes extract other patterns that would not correspond to the desired fractions (such as 4/7, which stemmed from 24/7).

Here we identify some messages that were missing their other half. These are rows which have iteration_id = 2/2 and count_username = 1. Given that these messages are incomplete, we decide not to try to join the messages anymore. What we do instead is assign a weight to each of the messages, which tells us how much it should contribute to the mean length.

We thus proceed to create the column weight. This is done by taking the denominator of the column iteration_id, and concatenating that with a nominator of 1. This way, an iteration_id of 2/3, 2/2, and 1/4 would get the weights 1/3, 1/2 and 1/4 respectively. The NAs of the iteration_id column will be assigned a weight of 1.

Now we proceed to clean the text. For this, we create a customized text_cleaner formula, which does the following:
Replace URLs (starting with "http://" or "https://") with an empty string.
Remove Twitter-like usernames (strings starting with '@').
Remove fractions in the format 'digit/digit' preceded by a space.
Remove any string starting with '*' and everything after it until the end of the line.
Replace newline characters ('\n') with a space.
Remove ellipsis ('...').
Remove two uppercase letters at the end of the string.
Unescape HTML to decode HTML entities (e.g., '&amp;' becomes '&;').
Replace the pattern '&;' with '&'.

Now that the text has been cleaned, we can proceed to count its length. After that, we transform the weight column to a float and multiply that by the length of the rows. Finally, we compute an average of the length for the whole dataset. This gives us an average length per reply of 63.3 in terms of characters, and of 11.22 in terms of words.




Question 2 - What type of links were referenced most often? URL links, phone numbers or direct messages?

To answer this question, we firstly create a separate dataset (df_q2) that contains the only column we need for the analysis (text).

Then, with the help of the regex expression r'https?://\S+' and .str.count, we count the number of URLs that start with either "http://" or "https://".

After that, we count the amount of times a phone number is present. For that, we use the following regex pattern: r'(\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})'. Its most important parts are:
Overall, it tries to find patterns of three digits (\d{3}), followed by three more digits, and then by four more digits.
Both \(?: and )? have the purpose of having optional parentheses for the first three digits of the number sequence. Eg: (800) 325 8224.
The [\s-]? matches an optional whitespace character (' ') or hyphen ('-'). The reason for this is that some of the numbers are put together with hyphens in between, while others are not (eg: 1-800-221-1212 and 800 221 1212).

Finally, we look for the direct messages. We firstly try to look for the “DM” sequence and add its counts per row to the column DM. However, after checking, we realize that the string “direct message” shows up as well. In some cases, this string has capital letters. For this reason, we use “(direct message)\s" to check if “direct message” is followed by a string, and then employ flags=re.IGNORECASE to make the search case-agnostic. These counts are added in the column DM_2. We then merge the counts of DM and DM_2 into DM_count.

Now that we have three columns with the respective counts of URL, phone numbers and DMs per row, we can proceed to add them up. This gives us the following results:
Total URL references: 59
Total phone number references: 77
Total direct messages references: 201

Thus, direct messages are referenced the most.




Question 3 - How many people should be on a social media customer service team? 

The first step to undertake here is to identify the workers. For that, we need to create a separate column with the initials of the workers. With this in mind, we firstly use .str.extract(r'\*(\w+)'). This extracts any group of characters that follows an asterisk.

However, this gives NAs in some instances. In order to try to find out what those missing values are, we split the newly-created dataset df_q3 in two parts. We will deal with the NAs in the part called df_worker_na.

The first step is to extract the missing values in worker_id with the help of the following regex expression: r'([A-Z]{2})$'. This will find all the initials at the end of the strings that were not preceded by the asterisk. After doing this, we concatenate back both df_worker_notna and df_worker_na.

Now that we have exhausted all the possibilities of extracting any string from the text column in order to obtain the worker_id, we will proceed to map the missing values. For that, we will have to create a mapping table where the following columns have been grouped: date, username and weight and worker_id. Note that we have chosen the largest possible combination of unique identifiers in order to make the mapping as accurate as possible.

This grouping step will only consider the rows where worker_id is not NA. For that, we again split the dataset in two, depending on whether worker_id is empty or not. Then we do the grouping, map the worker_id column, and merge the two datasets again. After checking the missing NAs again and realizing that only 4 NAs can be found, and that these are cases where no information on the worker_id is provided in the text column, we decide to move forward.

We decide to group by date and weekday and to count the unique number of workers that were present each day. As we can see in the table, the number of daily workers varies substantially, from 11 to 20. It seems that Tuesdays and Thursdays tend to employ more people, and that Fridays and Saturdays are less busy. For this reason, we decide to group by weekday and average the worker count found in the first grouping. This reduces the range a bit, from 11.5 to 17.5, and tells us that Tuesdays and Thursdays need indeed more people, whereas Fridays and Saturdays need less.

To sum up, having between 12 and 17 people in a social media customer service team seems to be enough to cover the demand. If it is possible, this number should be adjusted by weekday, given that the amount of work seems to vary from day to day.




Question 4 - How many social replies are reasonable for a customer service representative to handle?

To answer this question, we wanted to see how many responses every representative is handling on a daily basis, and then group that by worker in order to see the average amount of daily replies that each representative was handling.

After doing this, we noticed that the range of those average daily replies was quite high; it goes from 13.71 to 1. For this reason, we decided to find out how many days each of the workers was active. We could then filter out the workers that were the least active, given that they might be introducing some noise to the results.

To add the days_active column we first had to group by worker_ud and count the number of occurrences. We then mapped the days_active to the main dataframe. After filtering it to days_active >= 5, we still obtain very similar results.

Thus, to get more specific, we obtain the mean and the values of the 25th and 75th percentiles, which are 5.69, 4.38 and 7.33 respectively. This tells us that a customer service representative should handle between 5 and 7 social replies.

