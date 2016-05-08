
# Pump It Up! Data Mining the Water Table 
A [DrivenData.org competition](https://www.drivendata.org/competitions/7/page/25/) undertaken by Metis Data Science Fellows (Spring 2016):  
* https://github.com/ashirwad08)  
* https://github.com/ak29s12  
* https://github.com/knpatel401 
* https://github.com/jgondin  
* https://github.com/jcc-ne 
---  


# APPENDIX - PROJECT NOTES  

[iPython Notebook with investigations](./data_discovery.ipynb)

## Data Discovery
Overall, 59,400 Observations and 41 Features. Attempt to predict the operational status of water pumps in Tanzania.  

### Outcome Variable 
*status_group*, has 3 classes
* functional (~55%)
* non functional (~37%)
* functional needs repair (~7% imbalanced!) 
    * need oversampling/undersampling strategy

### Temporal Variables
* _date_recorded_: The date the row was entered
    * (extract new feature) year:
        * 2004 and 2002 readings constitute < 1% of total dataset
        * no clear trends with year and pump health prevalence
        * After checking the outcome group proportion by year data recorded, 2002 is skewing the proportions. __Suggest remove record with date_recorded year == 2002__.
* _construction_year_: Year the waterpoint was constructed
    * construction age suggests that older pumps go “non functional” more than newer ones
    * about 35% of the dataset is missing construction year. The pump health’s prevalence in these data points is pretty representative of the samples observed in other years, so we can’t discard these data points. 

### Continuous Variables
* _amount_tsh_: Total static head (amount water available to waterpoint)
    * About 70% “0” readings, but these do __not__ predict pump health outcome. It seems like these are “missing” readings.
    * Values very right skewed for dominant outcome classes
    * More _amount_tsh_ data points available for “functional” pumps
    * Might need an impute strategy here to populate by geographic location (assumption being that the amount of water available to a pump node should be more or less similar to water available to pumps that are close to it)
* _num_private_: no description available but it has numeric values (doesn’t seem categorical)
    * extremely sparse. 98.7% missing
    * values are very, very right skewed, for both dominant outcome classes
* _population_: Population around the well
    * 36% population values are either missing or true indicators (== 0). Since it wouldn’t make sense for a water pump to be in a non-populated area, we assume 36% of population readings are missing. Again, must figure out an impute strategy possibly driven by populations in close geographic locations.
    *  most “functional need repair” data points occur in lower population areas; higher population seems somewhat correlated to more functioning pumps

#### Location Variables
* _gps_height_: Altitude of the well
    * assuming “0” is ground level and not missing!
    * more wells at lower altitudes but can’t really tell if gps_height discerns well between classes  
* Latitude, Longitude, V gps_height, 3D plot confirms that there are more _"non functional"_ pumps at ground level and in higher altitudes to the South, and _"functional need repair"_ pumps at higher North-Western altitudes.  

---   


# APPENDIX - MISC  

## predict-water-pump-failure
* Team "Hans and Franz", competing in the Driven Data challenge https://www.drivendata.org/competitions/7/page/23/ 

* [see project planing/progress](https://htmlpreview.github.io/?https://raw.githubusercontent.com/https://github.com/ashirwad08/predict-water-pump-failure/master/organization/brainstorm_planing/Overview.html) 

