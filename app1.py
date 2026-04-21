{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e53c1e7e-cfb4-4a9d-bc1e-0ad9fe99b7d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Wine Quality: 6\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "model = joblib.load(\"winequalityN_model.pkl\")\n",
    "\n",
    "data = pd.DataFrame([[\"white\", 7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.00, 0.45, 8.8]],\n",
    "columns=[\n",
    "    \"type\", \"fixed acidity\", \"volatile acidity\", \"citric acid\",\n",
    "    \"residual sugar\", \"chlorides\", \"free sulfur dioxide\",\n",
    "    \"total sulfur dioxide\", \"density\", \"pH\", \"sulphates\", \"alcohol\"\n",
    "])\n",
    "\n",
    "# convert type\n",
    "data[\"type\"] = data[\"type\"].map({\"red\": 0, \"white\": 1})\n",
    "\n",
    "prediction = model.predict(data)\n",
    "print(\"Predicted Wine Quality:\", prediction[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40770624-d8c6-41c2-b1de-21b8a9c25414",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20f3f237-e940-4a97-857c-2eb22db28bf1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
