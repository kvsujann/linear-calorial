# app.py
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Устанавливаем неинтерактивный бэкенд для работы в серверной среде
import matplotlib.pyplot as plt
import io, base64, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

DATA_DIR = "data"

# ---------- Helpers ----------
def list_csv_files():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]

def find_column(df, candidates):
    """Find column name in df from list of candidates (case-insensitive & stripped)."""
    for c in candidates:
        if c in df.columns:
            return c
    # normalized
    cols_norm = {col.lower().replace(" ", "").replace("-", "").replace("_",""): col for col in df.columns}
    for c in candidates:
        key = c.lower().replace(" ", "").replace("-", "").replace("_","")
        if key in cols_norm:
            return cols_norm[key]
    return None

def load_and_prepare(path):
    raw = pd.read_csv(path)
    # try find names
    protein_col = find_column(raw, ['Protein (g)', 'Protein', 'protein_g', 'proteins', 'protein'])
    fats_col = find_column(raw, ['Fat (g)', 'Fat', 'fat_g', 'fats', 'fat'])
    carbs_col = find_column(raw, ['Carbohydrates (g)', 'Carbs', 'Carbohydrate', 'carbs', 'carbohydrate_g', 'carbohydrates'])
    cal_col = find_column(raw, ['Calories (kcal)', 'Calories', 'energy_kcal', 'kcal', 'calories', 'energy'])
    weight_col = find_column(raw, ['Weight (g)', 'Weight', 'weight_g', 'weight'])
    
    required_cols = [protein_col, fats_col, carbs_col, cal_col]
    if None in required_cols:
        missing = []
        if protein_col is None: missing.append('Protein')
        if fats_col is None: missing.append('Fat')
        if carbs_col is None: missing.append('Carbs')
        if cal_col is None: missing.append('Calories')
        raise ValueError(f"Не найдены колонки: {missing}. Проверь CSV или переименуй столбцы.")
    
    # Собираем только необходимые колонки
    columns_to_use = [protein_col, fats_col, carbs_col, cal_col]
    if weight_col: columns_to_use.append(weight_col)
    
    df = raw[columns_to_use].copy()
    df.columns = ['Protein', 'Fat', 'Carbs', 'Calories'] + [c for c in ['Weight'] if find_column(raw, [c]) is not None]
    
    # numeric convert + dropna
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    if df.shape[0] < 5:
        raise ValueError("После очистки в датасете меньше 5 валидных строк.")
    return df

def train_and_select_model(df):
    X = df.drop(columns=['Calories'])  # Используем все признаки, кроме целевой переменной
    y = df['Calories']
    # remove constant columns
    const_cols = [c for c in X.columns if X[c].var() == 0]
    X = X.drop(columns=const_cols)
    used_features = list(X.columns)
    if X.shape[1] == 0:
        raise ValueError("Нет признаков (все константы).")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = {
        'LinearRegression': make_pipeline(StandardScaler(), LinearRegression()),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42)
    }
    best = None
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        results[name] = {'model': m, 'r2': r2, 'mae': mae, 'mse': mse, 'y_test': y_test, 'y_pred': pred}
        if best is None or r2 > best[0]:
            best = (r2, name)
    best_name = best[1]
    best_entry = results[best_name]
    return {
        'best_model_name': best_name,
        'best_model': best_entry['model'],
        'metrics': {'r2': best_entry['r2'], 'mae': best_entry['mae'], 'mse': best_entry['mse']},
        'y_test': best_entry['y_test'],
        'y_pred': best_entry['y_pred'],
        'used_features': used_features,
        'const_removed': const_cols,
        'scaler': scaler
    }

def create_plot(y_actual, y_predicted, user_point=None):
    plt.figure(figsize=(6,5))
    plt.scatter(y_actual, y_predicted, color='dodgerblue', alpha=0.7, s=30, label='Predicted vs Actual')
    mn = min(y_actual.min(), np.min(y_predicted))
    mx = max(y_actual.max(), np.max(y_predicted))
    plt.plot([mn, mx],[mn, mx], 'r--', linewidth=2, label='Ideal Fit')
    if user_point is not None:
        plt.scatter([user_point],[user_point], color='green', s=90, edgecolor='k', label='Your prediction')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Actual vs Predicted Calories')
    plt.legend()
    plt.grid(alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return img_b64

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    files = list_csv_files()
    selected = request.form.get('dataset') or (files[0] if files else None)
    message = None
    preview_html = None
    plot_url = None
    predicted = None
    train_info = None

    if selected is None:
        message = "Положи CSV в папку `data/` и перезагрузите страницу."
        return render_template('index.html', files=files, message=message)

    try:
        df = load_and_prepare(os.path.join(DATA_DIR, selected))
        preview_html = df.head(10).to_html(classes='data', index=False)
        train_res = train_and_select_model(df)
        metrics = train_res['metrics']
        used_features = train_res['used_features']
        const_removed = train_res['const_removed']
        plot_url = create_plot(train_res['y_test'], train_res['y_pred'])
        train_info = {
            'best_model': train_res['best_model_name'],
            'metrics': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()},
            'used_features': used_features,
            'const_removed': const_removed,
            'rows': len(df)
        }

        if request.method == 'POST' and request.form.get('action') == 'predict':
            try:
                protein = float(request.form.get('protein', 0))
                fat = float(request.form.get('fat', 0))
                carbs = float(request.form.get('carbs', 0))
                weight = float(request.form.get('weight', 0))
            except ValueError:
                message = "Пожалуйста, введите числовые значения для всех полей."
                return render_template('index.html', files=files, selected=selected, message=message)
            vals = [0.0] * len(used_features)
            for i, c in enumerate(used_features):
                if c.lower().startswith('protein'): vals[i] = protein
                elif c.lower().startswith('fat'): vals[i] = fat
                elif c.lower().startswith('carb'): vals[i] = carbs
                elif c.lower().startswith('weight'): vals[i] = weight
            X_user = np.array(vals).reshape(1, -1)
            X_user_scaled = train_res['scaler'].transform(X_user)
            pred = train_res['best_model'].predict(X_user_scaled)[0]
            predicted = round(float(pred), 2)
            plot_url = create_plot(train_res['y_test'], train_res['y_pred'], user_point=predicted)

    except Exception as e:
        message = str(e)

    return render_template(
        'index.html',
        files=files,
        selected=selected,
        message=message,
        preview_html=preview_html,
        plot_url=plot_url,
        predicted=predicted,
        train_info=train_info
    )

if __name__ == "__main__":
    app.run(debug=True)