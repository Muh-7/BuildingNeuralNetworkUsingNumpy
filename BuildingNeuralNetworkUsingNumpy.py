import numpy as np

class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        
        # هون عم اهيء الاوزان و البايز لكل طبقة
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))


    # ضفت هون تابع تففعيل يلي خو الريلو  بيقارن القيمة مع الصفر  
    def relu(self, x):
        return np.maximum(0, x)


    # وعملناها عالباتشات Forward هون عملية ال
    def forward(self, X):
        
        # طبقة مخفية يلي هي الهيدن لاير
        z1 = X @ self.W1 + self.b1
        h1 = self.relu(z1)

        #  (h1)طبقة مخرجات وهون بكون عندي الدخل هو خرج الطبقة السابقة طبعااا
        z2 = h1 @ self.W2 + self.b2
        return z2


    # Prediction هسع بنيجي لتابع ال
    # يلي بالاساس عملنا كل هاللفة مشان ما نستدعيه مرات كثير مثل الطريقة السابقة وبالتالي بزيد عندي التكلفة بالحسابات و الموارد فطبقنا فكرة الباتشات 
    def predict(self, X):
    #بنفذلي الفوروورد
        logits = self.forward(X)
    #جيبلي الاحتمال الاكبر يلي نتج عندي
        return np.argmax(logits, axis=1)


    # لنعرف دقة شغلنا Accuracy حساب ال
    def accuracy(self, X, y_true):

        preds = self.predict(X)
        return np.mean(preds == y_true)



input_dim = 784
hidden_dim = 64
output_dim = 10  

model = NeuralNet(input_dim, hidden_dim, output_dim)

# دفعة حجمها 128 مثال
X_batch = np.random.randn(128, input_dim)

# :) عشوائي فقط للتجربة
y_batch = np.random.randint(0, 10, size=128)

# اطبعلي الباتشات المتوقعة
print("Predictions:", model.predict(X_batch))
#جيبلي الدقة والله يعطيك العافية
print("Accuracy:", model.accuracy(X_batch, y_batch))




argets = {
    1: [(0,0),(3,0)],
    2: [(0,3),(3,3)],
    3: [(2,1),(0,2)]
}