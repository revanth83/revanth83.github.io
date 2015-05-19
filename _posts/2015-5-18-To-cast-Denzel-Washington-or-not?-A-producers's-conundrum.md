---
layout: post
Title: Processing the MTA turnstile data!
---

Firstly, let's go over the client and the problem. My client is a reclusive producer named Charlie. He is the same Charlie from Charlie's Angels. So, we don't know much about him other than the fact that he is considering Denzel Washington as a lead in his next movie  and would like to know if casting Denzel would help generate more Gross Domestic Revenue than otherwise. I will step you through my approach in the following slides

<figure>
  <img src="{{ site.baseurl }}/images/denzelpic1.png">
  <figcaption>Fig1. - Denzel Washington in Training Day.</figcaption>
</figure>

<figure>
  <img src="{{ site.baseurl }}/images/charlie's_angels_pic2.png">
  <figcaption>Fig2. - Charlie's angels deal with me on his behalf.</figcaption>
</figure>

I relied on 2 data sources: boxofficemojo.com and wikipedia.com. I used BeautifulSoup package to scrape various quantities of interest such as Gross Domestic Revenue and Production budget, Date released etc. OF 15,000 movies fromboxofficemojo.com. Subsequently, using the same, I scraped movie list of Denzel Washington from WikiPedia. 

I scrutinized the movie data related to Denzel Washington to determine an appropriate date range for the analysis. For example, Denzel started his Film career in 1976; however, he simply acted as a street thug in the movie. I only wanted to consider Denzel's movies in which he played a significant role. His first notable role was in the movie License to kill (released in 1984).  It is different from Timothy Dalton's License to kill that was released in 1982.  

Subsequently, I performed statistical analysis using Statsmodels package to understand if the gross domestic revenue generated per unit budget spent is higher for movies that cast Denzel washington than those without. As you can observe from the slope of the best linear regression fit line associated with the data points, the presence of Denzel washington does not necessarily bring in more revenue.

<figure>
  <img src="{{ site.baseurl }}/images/regression_plot_pic3.png">
  <figcaption>Fig3. - The regression plot demonstrating that the gross revenue per unit budget is lower for
  Denzel's movies on an average than those of other Hollywood stars.</figcaption>
</figure>

If I had more time, I would carry out a more detailed analysis to understand the circumstances under which Denzel Washington brings in more revenue than an average Hollywood star. We need to segment the data further with respect to Genre/year/season to shed more light on this question. 



