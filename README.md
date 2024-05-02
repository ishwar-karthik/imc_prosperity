# IMC Prosperity Trading Challenge 2024
Algebros 4000's algorithms during IMC's Prosperity trading competition between 8 and 25 April 2024. 
Final place: ? of ~10,000 teams (Position not released yet, last position released was 553rd of 10,007 teams)

## Team Members:
- Ishwar Kalyan Karthik - LinkedIn: https://www.linkedin.com/in/ishwar-karthik-3b62a9253/
- Mountain Cheng - LinkedIn: https://www.linkedin.com/in/mountain-cheng/
- Henry Roskill - LinkedIn: https://www.linkedin.com/in/henry-roskill-317a65249/
- Edward Liu - LinkedIn: https://www.linkedin.com/in/ed-l-917896255/
- Jessica Pulford - LinkedIn: https://www.linkedin.com/in/jessica-pulford-49a2b220b/

## Final overall strategy (algorithmic trading)
In the "active_algo.py" file. 

### Split into 3 types of products
Stable, Volatile, Momentum. 

Idea:
- stable products: price has small fluctuations around a relatively constant value
- volatile products: price can fly around so much that we have to calculate a fair price for this product on the fly
- momentum products: there's a meaningful constant fair value over time, but the market price can fluctuate a lot around this.

In practice, we just assigned this stuff at the beginning, tested it on the data bottles they gave us, and tweaked the classification to try and get convincing profit on test data
(convincing as in, our position on the product fluctuates. So that we don't get profits by randomly holding a position and the prices happened to line up nicely.)

### General strategy for each of these 3 product types

#### Stable products
Our tactics for stable products were copied off last year's 2nd placed team, Stanford Cardinal (https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal).
(what they did for pearls). Just market-make & market-take around our constant value of 10k (that was mentioned in the video)

#### Volatile products

For volatile products, we used the VWAP metric (volume-weighted average price) to gauge what a fair price was. 
To avoid getting trigger-happy, we calculated all the past VWAPs and averaged them. 
The issues here are clear: a) the storage needed was high, and b) we want to weight recent data more heavily. 

To solve both of these issues (albeit crudely), we imposed a cutoff of 400 VWAP values to store at any one time.
However, the timestamp was quite small, and so we only stored a VWAP every 150 timestamps. 
So our 400 entries span a range of 400*150 = 45,000 timestamps, which meant that the past_vwap variable wasn't feeding us redundant information.

We tried having an exponential weight for these 400 values, but it turned out to be better to weight them equally
and then take the more conservative estimate of a "fair price" between the past_vwap and the current_vwap.

After accounting for tariffs, we modified the Stanford team's buying/selling conditions using these new parameters. 

#### Momentum products

It was fairly similar to the strategy for volatile products, except we tried to explicitly out-smart the opposition by being braver than them. 
We used the RSI metric to compare the market price with our own analysis of a fair price,
and once we thought there was a big difference we replaced our buying and selling prices with the market's most aggressive buy and sell orders plus/minus 1 (to ensure our order gets matched).

This RSI metric was calculated using the same list of historical VWAPs - we checked whether the trend line was going up or down. 
Essentially, if it's been going up a suspicious amount, we say it's a bubble of overbuying and short it, and vice versa if it's been going down.

Naturally, this method was very risky and we only used it for coconut coupons at the end (as it's a future & is thus deep into the speculative end of financial instruments).

### Product-specific modifications
#### Orchids
Orchid production is affected by weather patterns, and our trading algorithm has access to the latest weather data. Lower production means supply decreases and price increases. 
Thus, we added a small method to bump up our fair price estimate if production is low, and conversely bump it down if production is high.
The specific hyperparameters were tuned by hand after faffing about with test data. 

## Things we tried that didn't work/were worse than existing stuff

Using the MACD indicator - in active_algo_macd.py.
To be fair, we could have tuned some hyperparameters and seen if we could get a version to work for any of our products, but our training data was so limited that overfitting would be near-guaranteed if we did that. 

Edward suggested a variant of the RSI metric - create a histogram of the past VWAPs and see what "percentile" the current VWAP is at compared to those. 
However, the issue with this largely was that the "past VWAPs" are still recent in the big-picture. 
We could have had some sort of rough histogram of ALL the past VWAPs (rough to save on space/time) and tried something based on that, but we were already working heavily on the current algorithm and didn't want to risk much. 
