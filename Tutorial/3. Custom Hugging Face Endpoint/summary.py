# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import pipeline

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

ARTICLE = """
SPEAKER_00:  or in a coffee shop or the location that the job is set at, what would you rather do? 
SPEAKER_01:  Well, that really, really depends on the workplace, right? 
SPEAKER_00:  Because a lot of my jobs that I've had, like I worked at a factory on an assembly line, 
SPEAKER_01:  I'd rather work from home. 
SPEAKER_00:  Than that, yeah. 
SPEAKER_01:  Maybe just a standard office job. 
SPEAKER_01:  I kind of like the idea of I work one day at the office and I work the rest of the 
SPEAKER_00:  days at home like to get a little bit of that variety um but in general I think I would choose 
SPEAKER_01:  a workplace um just so because I like to be really active so like my job now I'm teaching children 
SPEAKER_00:  elementary kids so when I'm in a classroom, I got to be like really moving 
SPEAKER_01:  on top of it and basically on my A game. And at home, I don't get that same energy. I just 
SPEAKER_00:  kind of fall into like a... I don't... Maybe a rut is kind of too dramatic of a word, but 
SPEAKER_01:  I can more easily get into a rut at home. 
SPEAKER_00:  So you feel more motivated when you're on the location. If Dan had a summer job one 
SPEAKER_01:  time in high school, was it in high school? 
SPEAKER_00:  What job are you talking about? 
SPEAKER_01:  When you were working at Sony? 
SPEAKER_00:  That was college, yeah. 
SPEAKER_01:  Okay. A summer job where you just- 
SPEAKER_00:  That was a factory. 
SPEAKER_01: ... tapped a TV for 10 hours a day to check it. 
SPEAKER_00:  Or move it. There was some variety. 
SPEAKER_01:  Sometimes we got to pick up the TV and put it on the table. 
SPEAKER_00:  Yes. 
SPEAKER_01:  That was a job- 10 hours. 
SPEAKER_00:  Every day. 
SPEAKER_01:  Yeah. 
SPEAKER_00:  That's a job that really makes you grateful for the other jobs that you have in life. 
SPEAKER_01:  But this type of job, first of all, you'd rather not do, but also that's, if that's your comparison 
SPEAKER_00:  point, you would rather work at home. But in general, working in the workplace is better for 
SPEAKER_01:  you. Yes. For me, it's, it's better. And I mean, in our scenario, we worked together from home for 
SPEAKER_00:  a while and because both of us were working at home, it really turned into like our entire life 
SPEAKER_01:  is just in this house. Yeah. Especially because we had 
SPEAKER_00:  young kids and we're working from home. It was really hard to have a community and get out 
SPEAKER_01:  because to get out with small children is already tricky and you're working. So I think you working 
SPEAKER_00:  at the school now just adds another wonderful layer to our family where we have a community, 
SPEAKER_01:  adds another wonderful layer to our family where we have a community, you have coworkers, 
SPEAKER_00:  there's like more, yeah, it's also a nice community. It's a nice workplace. Some workplaces are not nice. That makes a big difference. Yeah, for sure. I think for me, because I work for 
SPEAKER_01:  myself, I have my own business teaching you English, speak English with Vanessa. I'm going to answer neither. 
SPEAKER_00:  I don't want to work at home and I also don't want to work in a workplace. 
SPEAKER_01:  This is my ideal scenario. 
SPEAKER_00:  Are you ready? 
SPEAKER_01:  She wants to be totally independent. 
SPEAKER_01:  She wants to work in space. 
SPEAKER_00:  I want to work in a castle overlooking a kingdom. 
SPEAKER_00:  No. 
SPEAKER_00:  I want to... and I can probably do this in another couple of years, work for my 
SPEAKER_00:  business. Great. But not at home. A couple last year when I was pregnant with my baby, I worked 
SPEAKER_00:  at a coworking space and these exist all around the world. Maybe you've heard of them, maybe you 
SPEAKER_00:  haven't, but it's like a really fun office. So like they try to make it cool because they want you to go 
SPEAKER_00:  there and everyone is working just on their laptops for their own jobs. Freelancers. Freelancers. Or 
SPEAKER_01:  maybe they're working for people like you, various companies, or maybe they run a company 
SPEAKER_00:  and there's events, there's food, there's food trucks, there's things going on. 
SPEAKER_01:  It's the fun place to be. 
SPEAKER_00:  And this is, I think, my ideal, where I'm working at the job that I want to work at, 
SPEAKER_01:  being your English teacher, and also in a location where I'm around other people. 
SPEAKER_00:  Working from home is so convenient with small children. My baby's napping in the other 
SPEAKER_00:  room right now. I can nurse her. I can help my kids. I can do those types of things. But when 
SPEAKER_01:  they're a little bit older, I'm out. I want to go and be out in a co-working space in this type of 
SPEAKER_00:  environment I think is really healthy for me. If you couldn't tell, she is a people person. 
SPEAKER_01:  I like to be around other people, 
SPEAKER_00:  but I know some people can work, for example, 
SPEAKER_01:  like in a coffee shop or this busy environment. 
SPEAKER_01:  I cannot. 
SPEAKER_00:  If it's an absolute necessity, like a requirement. 
SPEAKER_01:  Public place. 
SPEAKER_00:  We have no internet at home 
SPEAKER_00:  and I have to get something done. 
SPEAKER_00:  Yes, I can go to a coffee shop and get some work done, 
SPEAKER_00:  but that is not my ideal environment. 
SPEAKER_01:  I'm not productive in that environment either. 
SPEAKER_00:  Yeah, some people are, some people thrive with the chaos 
SPEAKER_01:  and lots of stuff going on, but that's not my jam. 
SPEAKER_00:  All right, so now are you ready for a quiz? 
SPEAKER_01:  Oh, we're to the quiz? 
SPEAKER_00:  The next section is a quiz. 
SPEAKER_01:  Yes. All right, these are some, 
"""


print(summarizer(ARTICLE, max_length=5000, min_length=30, do_sample=False))