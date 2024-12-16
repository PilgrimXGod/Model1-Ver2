let model;
let vocabulary;
const categories = ['World', 'Sports', 'Business', 'Science/Tech'];

async function loadModel() {
    try {
        model = await tf.loadLayersModel('tfjs_news_model/model.json');
        const response = await fetch('tfjs_news_model/model_config.json');
        const config = await response.json();
        vocabulary = new Map(config.vocabulary.map((word, i) => [word, i]));
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

function preprocessText(text) {
    // Токенізація та векторизація тексту
    const tokens = text.toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .slice(0, 128);

    const sequence = tokens.map(token => vocabulary.get(token) || 0);
    while (sequence.length < 128) {
        sequence.push(0);
    }

    return tf.tensor2d([sequence]);
}

async function classifyNews() {
    const textArea = document.getElementById('newsInput');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const categoryResult = document.getElementById('categoryResult');
    const confidenceResults = document.getElementById('confidenceResults');

    if (!textArea.value.trim()) {
        alert('Please enter some text');
        return;
    }

    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const inputTensor = preprocessText(textArea.value);
        const predictions = await model.predict(inputTensor).data();

        // Знаходження категорії з найвищою ймовірністю
        const maxIndex = predictions.indexOf(Math.max(...predictions));
        const maxConfidence = predictions[maxIndex];

        // Відображення результатів
        categoryResult.innerHTML = `<p><strong>Category:</strong> ${categories[maxIndex]}</p>`;

        // Відображення всіх ймовірностей
        confidenceResults.innerHTML = categories.map((category, i) => `
                    <p>${category}: ${(predictions[i] * 100).toFixed(2)}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${predictions[i] * 100}%"></div>
                    </div>
                `).join('');

        result.style.display = 'block';
    } catch (error) {
        console.error('Error during classification:', error);
        categoryResult.innerHTML = 'Error during classification';
    } finally {
        loading.style.display = 'none';
    }
}

// Завантаження моделі при запуску
loadModel();