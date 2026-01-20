DATA DESCRIPTION.

Chonnam National University Data Sharing.zip

• A training set was created for patients admitted to Chonnam National University Hospital in Hwasun from March 2017 to February 2019. A test set was constructed for patients admitted to Chonnam National University Hospital in Hakdong from January 2019 to April 2019. This model was then developed.

• Event time – Cardiac arrest/endotracheal intubation during hospitalization.

• Detection time – The time at which the medical staff is believed to have recognized the patient's abnormality.

• Test group (Hwasun abn): Patients admitted to Chonnam National University Hospital in Hwasun from March 2017 to February 2019 who experienced cardiac arrest or underwent in-hospital endotracheal intubation.
• Control group (Hwasun NL): Patients admitted to the Department of Internal Medicine (Gastroenterology, Cardiology, Respiratory Medicine, Nephrology, Endocrinology, Hematology, Oncology, Infectious Diseases, and Allergy) from March 2017 to February 2019, who were hospitalized for 5 to 31 days and did not undergo cardiopulmonary resuscitation or endotracheal intubation during their hospitalization.
• Test group (Hakdong ABN): Patients admitted to the Department of Internal Medicine (Gastroenterology, Cardiology, Respiratory Medicine, Nephrology, Endocrinology, Hematology, Oncology, Infectious Diseases, Allergy, and Rheumatology) at Chonnam National University Hospital (Hakdong) from January 2019 to April 2019, who received cardiopulmonary resuscitation and endotracheal intubation. • Test Control Group (Hakdong NL): Patients admitted to the Departments of Internal Medicine (Gastroenterology, Cardiology, Pulmonary Medicine, Nephrology, Endocrinology, Hematology, Oncology, Infectious Disease, Allergy, and Rheumatology) at Chonnam National University Hospital (Hakdong) from January 2019 to April 2019 who underwent cardiopulmonary resuscitation or endotracheal intubation.

• Hwasun/Hakdong Normal Group: Excluding those with a "death without event" category – Hwasun: Category 1, Hakdong: Category 2
• Abnormal Group: On the day of admission / 24 hours prior to event / 24 hours prior to detection.
◦ Even if event and detection lab data overlap, they are extracted as separate data.
◦ If there are two or more overlapping items (e.g., two CBCs in one day), the lab result closest to the corresponding time is used.
◦ Separate into Admission, Detection, and Event to confirm the blood collection time. Data structured to allow for:
• Normal group: Lab on the day of admission / 5 days later or at discharge
◦ If only a lab was available on the day of admission, only the lab on the day of admission was used.
◦ If no lab was available on the corresponding date, a lab performed between 4 and 10 days after admission was used.

_10yr.zip
• Test group: Patients admitted to Chonnam National University Hospital, Hwasun, from March 2009 to February 2019 who underwent cardiac arrest and in-hospital endotracheal intubation.

• Control group: Patients in the departments of Internal Medicine (Gastroenterology, Cardiology, Respiratory Medicine, Nephrology, Endocrinology, Hematology, Oncology, Infectious Disease, and Allergy) from March 2009 to February 2019 who were hospitalized for 5 to 31 days and did not undergo CPR or endotracheal intubation during their hospitalization.
• Test group (Hakdong abn): Patients admitted to the Department of Internal Medicine (Department of Gastroenterology, Department of Cardiology, Department of Respiratory Medicine, Department of Nephrology, Department of Endocrinology, Department of Hematology, Department of Oncology, Department of Infectious Diseases, Department of Allergy, Department of Rheumatology) of Chonnam National University Hospital (Hakdong) from January 2019 to April 2019, including those who underwent cardiopulmonary resuscitation and endotracheal intubation. • Test Control Group (Hakdong NL): Patients admitted to the Departments of Internal Medicine (Gastroenterology, Cardiology, Respiratory Medicine, Nephrology, Endocrinology, Hematology, Oncology, Infectious Disease, Allergy, and Rheumatology) at Chonnam National University Hospital (Hakdong) from January 2019 to April 2019 who underwent cardiopulmonary resuscitation or endotracheal intubation.

• Patients who died without an event in the normal group should be excluded. A coded value of 1 indicates an in-hospital death.

Data Sampling
• Among patients with an event in Hwasun and Hakdong, 100 (sampling 1) or 200 (sampling 2) were randomly selected, labeled, and machine learning was performed.