# Import Library atau Package
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Init Flask
app = Flask(__name__)

# Simpan data pada dataset ke dalam df
df = pd.read_csv('buy_computer.csv')

# Drop column yang tidak terpakai
df.drop('id', axis = 1, inplace = True)

# Simpan data attribute ke dalam variable x dan y
# Variable X berisi attribute (age, income, student, credit_rating)
x = df.iloc[:,:-1].values
# Variable y berisi attribute (Buy_Computer)
y = df.iloc[:,-1].values

# Init LabelEncoder untuk mengubah string menjadi number
enc = LabelEncoder()

# Ubah value pada masing masing attribute dari string menjadi angka
x[:,0] = enc.fit_transform(x[:,0]) # Attribute age
x[:,1] = enc.fit_transform(x[:,1]) # Attribute income
x[:,2] = enc.fit_transform(x[:,2]) # Attribute student
x[:,3] = enc.fit_transform(x[:,3]) # Attribute credit_rating
y = enc.fit_transform(y) # Attribute Buy_Computer

# Membuat model Decision Tree
model = DecisionTreeClassifier()

# Melakukan pelatihan model terhadap data
model.fit(x, y)

# Route beranda atau halaman awal
@app.route("/")
# Function index
def index():
    # Render view
    return render_template('index.html')

# Route prediction untuk passing data dari view menggunakan method POST
@app.route('/prediction', methods=['POST'])
# Function prediction
def prediction():
    # Simpan data dari masukan user
    age             = int(request.form['age'])
    income          = int(request.form['income'])
    student         = int(request.form['student'])
    credit_rating   = int(request.form['credit_rating'])
    
    # Lakukan prediksi menggunakan model yang sudah dilatih dengan data dari masukan user
    predicted = model.predict([[age, income, student, credit_rating]])

    # Ubah value age dari number menjadi string
    if age == 0:
        age = "Middle Age"
    elif age == 1:
        age = "Senior"
    elif age == 2:
        age = "Youth"

    # Ubah value income dari number menjadi string
    if income == 0:
        income = "High"
    elif income == 1:
        income = "Low"
    elif income == 2:
        income = "Medium"

    # Ubah value student dari number menjadi string
    student = "Yes" if student else "No"

    # Ubah value credit_rating dari number menjadi string
    credit_rating = "Fair" if credit_rating else "Excellent"

    # Ubah value hasil prediksi dari number menjadi string
    predicted = "Yes" if predicted else "No"

    # Render template dan passing data hasil prediksi ke dalam view
    return render_template('index.html', age = age, income = income, student = student, credit_rating = credit_rating, predicted = predicted)

# Driver
if __name__ == '__main__':
    app.run(debug=True)