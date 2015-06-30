---
layout: post
Title: Processing the MTA turnstile data!
---
<style>
   img {
       display: block;
       margin: auto;
   }
</style>

**Problem statement**: Provide a brief report to the CEO of Samsung that includes only critical information regarding reviews of Samsung’s products retailed on Amazon.com.



The objective of this work is to develop a tool that could provide useful information to the CEO using smarter and faster analysis of large amounts of data.

<figure>
  <img src="{{ site.baseurl }}/images/Slide1.png">
  <figcaption style="text-align:center;">Fig 1. Examples of Samsung's electronics products.</figcaption>
</figure>

<figure>
  <img src="{{ site.baseurl }}/images/Slide2.png">
  <figcaption style="text-align:center;">Fig 2. Essence of Business Intelligence.</figcaption>
</figure> 

Figure 3 illustrates the overall project plan. More than a quarter million reviews have been written regarding the 1000 products that Samsung sells on Amazon.com. In case, the chief executive would like to know whether any interesting trend was observed in the reviews related to any product and the reasons for such trends. 

<figure>
  <img src="{{ site.baseurl }}/images/Slide3.png">
  <figcaption style="text-align:center;">Fig 3. Overview of the project and the list of libraries used.</figcaption>
</figure> 


Firstly, I define a sharpness index that records the number of spikes in the variation of average monthly rating. I also keep track of the total number of monthly reviews with respect to time to ensure that the observed variation in rating is an actual trend and not dominated by outliers. Based on the aforementioned criteria, I short list 20 products that have had an unusual degree of variation (either positive or negative) in their ratings. Subsequently, I plot the curves of variation in average rating per month and total number of reviews per month to obtain further clues regarding the trends in the reviews related to the short listed products. Once, I confirm that there is an interesting trend in the reviews related to the product, I carry out topic modeling of relevant reviews
(documents) using latent dirichlet modeling to identify hidden topics and summarise them by identifying the document that is closest to the centroid of the documents dominated by a particular topc in the LDA space. 

The plots of average rating and total number of monthly reviews for a particular product of interest are shown in Fig. 4 and Fig. 5. As it can be observed, the average rating undergoes plenty of ups and downs. But towards during 2013-2014 of the curve, i.e., more recent months, the values are generally lower than 3.5 on a more consistent basis. 

<figure>
  <img src="{{ site.baseurl }}/images/Slide4.png">
  <figcaption style="text-align:center;">Fig 4. Variation in average rating for Samsung galaxy detachable Multi-Travel Charger.</figcaption>
</figure> 

We have very few reviews per month at beginning between 2011-2012, where as there are many more reviews between 2013 and 2014. The average monthly rating associated with months that have a low number of reviewers is not trustworthy as these values are dominated by outliers such as people who tend to have extreme opinions get a greater weight during those months. However, there is a general increase in the number of reviewers between 2013-2014; and it coincides with the period where the average rating has been below consistently 3.5. To understand the reasons, I carry out topic modeling of the reviews of this product. The results of topic modeling
are available at the following [web-link](http://revanth83.github.io/images/pyldavis_koj_final_tp_1_5.html#topic=4&lambda=0.4&term=).

<figure>
  <img src="{{ site.baseurl }}/images/Slide5.png">
  <figcaption style="text-align:center;">Fig 5. Variation in total number of reviews per month for Samsung galaxy detachable Multi-Travel Charger.</figcaption>
</figure>


<figure>
  <img src="{{ site.baseurl }}/images/Slide6.png">
  <figcaption style="text-align:center;">Fig 6. Topic modeling overview.</figcaption>
</figure> 

From the summary of reviews of the detachable Multi-Travel Charger it can be observed that that 20.7% documents have a highest weightage of topic 1 in them and the representative document corresponding to such reviews is present in the third column of the first row. This is the type of summary that provides useful insights to CEO.  

<figure>
  <img src="{{ site.baseurl }}/images/Slide7.png">
  <figcaption style="text-align:center;">Fig 7. Summary of relevant reviews for 
  Samsung galaxy detachable Multi-Travel Charger.</figcaption>
</figure> 
