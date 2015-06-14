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

My fourth data science project at Metis is about sentiment analysis of tweets related to Apple Watch.
I used the twitter streaming API to pull 5000 tweets
that are tagged with #Applewatch. Subsequently, I carried
out sentiment analysis on each tweet using Text Blob python package.

<figure>
  <img src="{{ site.baseurl }}/images/fletch_polarity.png">
  <figcaption style="text-align:center;">Fig1. Polarity histogram of twitter data related to Apple watch.</figcaption>
</figure>

<figure>
  <img src="{{ site.baseurl }}/images/flecth_subjectivity.png">
  <figcaption style="text-align:center;">Fig2. Subjectivity histogram of twitter data related to Apple watch.</figcaption>
</figure> 
