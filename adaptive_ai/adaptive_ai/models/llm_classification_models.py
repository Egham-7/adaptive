from enum import Enum

from pydantic import BaseModel, Field


class DomainType(str, Enum):
    ADULT = "Adult"
    ARTS_AND_ENTERTAINMENT = "Arts_and_Entertainment"
    AUTOS_AND_VEHICLES = "Autos_and_Vehicles"
    BEAUTY_AND_FITNESS = "Beauty_and_Fitness"
    BOOKS_AND_LITERATURE = "Books_and_Literature"
    BUSINESS_AND_INDUSTRIAL = "Business_and_Industrial"
    COMPUTERS_AND_ELECTRONICS = "Computers_and_Electronics"
    FINANCE = "Finance"
    FOOD_AND_DRINK = "Food_and_Drink"
    GAMES = "Games"
    HEALTH = "Health"
    HOBBIES_AND_LEISURE = "Hobbies_and_Leisure"
    HOME_AND_GARDEN = "Home_and_Garden"
    INTERNET_AND_TELECOM = "Internet_and_Telecom"
    JOBS_AND_EDUCATION = "Jobs_and_Education"
    LAW_AND_GOVERNMENT = "Law_and_Government"
    NEWS = "News"
    ONLINE_COMMUNITIES = "Online_Communities"
    PEOPLE_AND_SOCIETY = "People_and_Society"
    PETS_AND_ANIMALS = "Pets_and_Animals"
    REAL_ESTATE = "Real_Estate"
    REFERENCE = "Reference"
    SCIENCE = "Science"
    SENSITIVE_SUBJECTS = "Sensitive_Subjects"
    SHOPPING = "Shopping"
    SPORTS = "Sports"
    TRAVEL_AND_TRANSPORTATION = "Travel_and_Transportation"
    OTHERDOMAINS = "Otherdomains"


class DomainClassificationResult(BaseModel):
    domain: DomainType = Field(description="Primary classified domain")
    confidence: float = Field(
        description="Confidence score for domain classification (0.0-1.0)"
    )
    domain_probabilities: dict[str, float] = Field(
        description="Probability scores for all domains"
    )


class ClassificationResult(BaseModel):
    task_type_1: list[str] = Field(alias="task_type_1")
    task_type_2: list[str] = Field(alias="task_type_2")
    task_type_prob: list[float] = Field(alias="task_type_prob")
    creativity_scope: list[float] = Field(alias="creativity_scope")
    reasoning: list[float] = Field(alias="reasoning")
    contextual_knowledge: list[float] = Field(alias="contextual_knowledge")
    prompt_complexity_score: list[float] = Field(alias="prompt_complexity_score")
    domain_knowledge: list[float] = Field(alias="domain_knowledge")
    number_of_few_shots: list[float] = Field(alias="number_of_few_shots")
    no_label_reason: list[float] = Field(alias="no_label_reason")
    constraint_ct: list[float] = Field(alias="constraint_ct")


class EnhancedClassificationResult(BaseModel):
    task_classification: ClassificationResult = Field(
        description="Task-based classification results"
    )
    domain_classification: DomainClassificationResult = Field(
        description="Domain-based classification results"
    )
