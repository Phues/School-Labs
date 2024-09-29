from _widgets import *

def Quiz1():   
    Q = create_multipleChoice_widget('Q1: Which is correct about the following statments on data science?',
                                        ['Data Science relies on software development processes + data analysis techniques ⇒ to build accurate artificial intelligence and machine learning models capable of extracting useful data and predicting the future patterns and behaviours',
                                         'Data science ⇒ the study of data and it is used  to develop methods to store, record, and analyse data to effectively get useful information.',
                                         'Data science ⇒  a combination of different techniques and theories taken from many fields such as Math and statistics, Computer Science and Domains/Business Knowledge.',
                                         'All of the above'],
                                      'All of the above')
    

    display(Q)

def Quiz2():   
    Q = create_multipleChoice_widget('Q2: Which of the following jobs was called by the Harvard Business Review the sexiest job of the 21st century?',
                                        ['Data Mining Specialist',
                                         'Data Analytics Specialist',
                                         'Data Scientist',
                                         'Big Data Specialist',
                                         'All of the above'],
                                      'Data Scientist')
    

    display(Q)

def Quiz3():   
    Q = create_multipleChoice_widget('Q3: ________ published  an interesting  paper describing a new field, "data science," which expands on data analysis.',
                                        ['Data Mining Specialist',
                                         'Mr. Jeff Wu',
                                         'Mr. Jeff Hammerbacher',
                                         'Mr. William Cleveland',
                                         'Mr. DJ Patil'],
                                      'Mr. William Cleveland')
    
    display(Q)
    
def Quiz4():   
    Q = create_multipleChoice_widget('Q4: Who is a data scientist?',
                                        ['Mathematician',
                                         'Statistician',
                                         'Software programmer',
                                         'All of the above'],
                                      'All of the above')
    

    display(Q)
    

def Quiz5(): 
    Q = create_multipleChoice_widget('Q5: Which of the following is one of the key data science skill?',
                                        ['Statistics',
                                         'Machine learning',                                         
                                         'Data cleaning',
                                         'All of the above'],
                                      'All of the above')
    
    display(Q)

def Quiz6(): 
    Q = create_multipleChoice_widget('Q6: Why Machine Learning in Data Science?',
                                        ['For Visualization',
                                         'For Prediction',
                                         'For Cleaning',
                                         'All of the above'],
                                      'For Prediction')
    
    display(Q)  

def Quiz7(): 
    Q = create_multipleChoice_widget('Q7: Data Analysis is a key aspects in',
                                        ['(a) Big Data',
                                         '(b) Business Intelligence',
                                         '(c) Data Analytics',
                                         '(d) Data Science',
                                         '(a), (c), and (d)',
                                         '(b), (c), and (d)'],
                                      '(b), (c), and (d)')
    
    display(Q) 

def Quiz8(): 
    Q = create_multipleChoice_widget('Q8: Business Intelligence and Big Data do not dail with actuall data:',
                                        ['True',
                                         'False',
                                         ],
                                   'False')

def Quiz9(): 
    Q = create_multipleChoice_widget('Q9: Data Analytics and Data Science may dail with _________:',
                                        ['Actual (Current) Data',
                                         'Future Data',
                                         'Both'
                                         ],
                                      'Both')

    display(Q) 


def Quiz10(): 
    Q = create_multipleChoice_widget('Q10: Types of data in Data Science can be  _________ :',
                                        ['Table',
                                         'Image',
                                         'Audio',
                                         'Text',
                                         'All of the above'
                                         ],
                                      'All of the above')

    display(Q)


def Quiz11():   
    Q = create_multipleChoice_widget('Q11: Which of the following are examples of unstructured data?',
                                        ['(a) Facebook Images',
                                         '(b) Tweeter Feeds',
                                         '(c) Table "Employee(emp_full_name, emp_hire_date, emp_dept, emp_salary)" in a relational database',
                                         '(a) and (b)',
                                         '(a) and (c)',
                                         '(a), (b), and (c)',
                                         'None of the Above'],
                                      '(a) and (b)')
    
    display(Q)


def Quiz12(): 
    Q = create_multipleChoice_widget('Q12: Given you have a data in table format with "House Area", "# Bedrooms", "# Accessible Facades", "Distance To Beach", and "Price" as columns.  What would be your possible input features if you are asked to predict the house price?',
                                        ['(a) House Area',
                                         '(b) # Bedrooms',
                                         '(c) # Accessible Facades',
                                         '(d) Distance To Beach',                                         
                                         'Any combination of (a), (b), (c), and (d)',
                                         '(a), (b), (c), and (d)',
                                         'One of the above'
                                         ],
                                      'One of the above')

    display(Q) 

def Quiz13(): 
    Q = create_multipleChoice_widget('Q13: A science project management can be approached by following a well-define stategy. Which one is the widely used strategy?',
                                        [
                                         'SCRUM',
                                         'TDSP',
                                         'KANBAN', 
                                         'CRISP-DM',                                        
                                         'SEMMA',
                                         'None of the above'
                                         ],
                                      'CRISP-DM')

    display(Q)

def Quiz14(): 
    Q = create_multipleChoice_widget('Q14: According to CRISP-DM Data Science Methodology, What is the  first part of your data science journey?',
                                        [
                                         'Model Evaluation',
                                         'Data Understanding',
                                         'Business Understanding', 
                                         'Data Preparation',                                        
                                         'Modeling',
                                         'Model Deployment'
                                         ],
                                      'Business Understanding')

    display(Q)


def Quiz15(): 
    Q = create_multipleChoice_widget('Q15: Which of the following tasks are true about Data Preparation Phase according to CRISP-DM Data Science Methodology?',
                                        [
                                         '(a) Data Ingestion',
                                         '(b) Data Selection',
                                         '(c) Data Cleaning', 
                                         '(d) Data Validation',                                        
                                         '(e) Modeling Techniques Selection',
                                         '(f) Model Monitoring & Maintainance',
                                         '(a) and (c)',
                                         '(b) and (d)',
                                         '(b) and (c)',
                                         '(a), (b), (c), and (d)',
                                         ],
                                      '(b) and (c)')

    display(Q)  

Quiz1()
Quiz2()
Quiz3()
Quiz4()
Quiz5()
Quiz6()
Quiz7()
Quiz8()
Quiz9()
Quiz10()
Quiz11()
Quiz12()
Quiz13()
Quiz14()
Quiz15()