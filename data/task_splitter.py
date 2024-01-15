import pandas as pd
from sklearn.model_selection import train_test_split

# classes
CLASS_COLUMNS = ['hemorrhage', 'vascular_occlusion','diabetic_retinopathy',
                 'macular_edema', 'scar', 'nevus', 'amd', 
                 'hypertensive_retinopathy', 'drusens', 
                 'myopic_fundus', 'increased_cup_disc']

def main():
    brset = pd.read_csv("data/labels.csv")

    #Drop unused columns
    brset = brset.drop(columns=['camera', 'patient_age', 'comorbidities',
       'diabetes_time_y', 'insuline', 'patient_sex', 'diabetes',
       'nationality', 'optic_disc', 'vessels', 'macula', 'DR_SDRG', 'DR_ICDR',
       'focus', 'iluminaton', 'image_field', 'artifacts'])

    brset = brset[brset["quality"]=="Adequate"] # Drop low quality data
    brset = brset.drop_duplicates(subset = ['patient_id', 'exam_eye']) # Drop repeated pictures for same eye

    # Sample 880 unconditioned (healthy) pictures

    healthy = brset[(brset.diabetic_retinopathy == 0) & (brset.macular_edema == 0) & (brset.scar == 0) & (brset.nevus == 0) & 
                    (brset.amd == 0) & (brset.vascular_occlusion == 0) & (brset.hypertensive_retinopathy == 0) & (brset.drusens == 0) &
                    (brset.hemorrhage == 0) & (brset.myopic_fundus == 0) & (brset.increased_cup_disc == 0) & (brset.other == 0) &
                    (brset.retinal_detachment == 0)].sample(n=880, random_state = 42)
    
    brset = brset[~brset.index.isin(healthy.index)]

    # Sample 80 positive samples from hemorrhage and vascular_occlusion 

    tasks = []

    hem = brset[ (brset.vascular_occlusion == 0) & (brset.hemorrhage == 1)]
    vo = brset[ (brset.vascular_occlusion == 1) & (brset.hemorrhage == 0)]
    intersec = brset[ (brset.vascular_occlusion == 1) & (brset.hemorrhage == 1)]

    hem = pd.concat([hem, intersec.iloc[0:8]])
    vo = pd.concat([vo, intersec.iloc[8:13]])
    brset = brset[~brset.index.isin(hem.index)]
    brset = brset[~brset.index.isin(vo.index)]

    tasks.append(('hemorrhage', pd.concat([hem, healthy.iloc[0:80]])))
    tasks.append(('vascular_occlusion', pd.concat([vo, healthy.iloc[80:160]])))

    # Sample 80 positive samples from the other tasks

    i = 0
    for condition in CLASS_COLUMNS[2:]:
        df = brset[brset[condition]==1]
        if len(df) >= 80:
            task = df.sample(n=80, random_state=6)
        else:
            task = df
        brset = brset[~brset.index.isin(task.index)]
        tasks.append((condition, pd.concat([task, healthy.iloc[160+80*i:160+80*(i+1)]])))
        i+=1

    for task,df in tasks:
        print(task, len(df))

    print()

    print("CHECKING REPEATED ROWS")
    df = pd.concat(task[1] for task in tasks)
    print(df.duplicated().any())
    print()

    # Splits for each class

    for task, df in tasks:
        shot_40, test = train_test_split(df, test_size=80, train_size=80, stratify=df[task])
        shot_20, _ = train_test_split(shot_40, train_size=40, stratify=shot_40[task])
        shot_10, _ = train_test_split(shot_20, train_size=20, stratify=shot_20[task])

        df['split'] = None

        df.loc[df.index.isin(test.index), 'split'] = "test"
        
        df.loc[df.index.isin(shot_40.index), 'split'] = "40-shot"

        df.loc[df.index.isin(shot_20.index), 'split'] = "20-shot"

        df.loc[df.index.isin(shot_10.index), 'split'] = "10-shot"


        assert (len(df[df.split == 'train'])) == 2 * (len(df[(df.split == 'train') & (df[task] == 1)]))
        assert (len(df[df.split == '40-shot'])) == 2 * (len(df[(df.split == '40-shot') & (df[task] == 1)]))
        assert (len(df[df.split == '20-shot'])) ==  2 * (len(df[(df.split == '20-shot') & (df[task] == 1)]))
        assert (len(df[df.split == '10-shot'])) == 2 * (len(df[(df.split == '10-shot') & (df[task] == 1)]))
        
        df.to_csv(f"data/{task}.csv")

    return 
    

if __name__ == "__main__":
    main()