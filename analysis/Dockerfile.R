FROM rstudio/r-base:4.0.4-bionic

COPY . . 

RUN R -e "install.packages('methods',dependencies=TRUE, repos='http://cran.rstudio.com/')"

RUN install2.r --error \
	methods \
	tidyverse \
	# other libraries I want 
	plumber
