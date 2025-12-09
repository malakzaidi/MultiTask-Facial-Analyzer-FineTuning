// Configuration
const API_ENDPOINTS = {
    upload: '/upload',
    predict: '/api/predict',
    modelInfo: '/api/model_info'
};

// Éléments DOM
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const attributesGrid = document.getElementById('attributesGrid');
const analyzedImage = document.getElementById('analyzedImage');
const processingTime = document.getElementById('processingTime');
const detectedCount = document.getElementById('detectedCount');
const totalCount = document.getElementById('totalCount');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadReportBtn = document.getElementById('downloadReportBtn');

// Variables globales
let currentImageFile = null;

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    totalCount.textContent = 8;
    setupEventListeners();
});

// Configuration des événements
function setupEventListeners() {
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Click sur la zone d'upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // Changement de fichier
    fileInput.addEventListener('change', handleFileSelect);

    // Boutons
    analyzeBtn.addEventListener('click', analyzeImage);
    clearBtn.addEventListener('click', clearImage);
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    downloadReportBtn.addEventListener('click', downloadReport);
}

// Gestion du drag and drop
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
}

// Gestion de la sélection de fichier
function handleFileSelect(e) {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    // Vérifier le type de fichier
    if (!file.type.match('image.*')) {
        showAlert('Veuillez sélectionner une image valide (JPG, PNG)', 'danger');
        return;
    }

    // Vérifier la taille
    if (file.size > 16 * 1024 * 1024) {
        showAlert('L\'image est trop volumineuse (max 16MB)', 'danger');
        return;
    }

    currentImageFile = file;

    // Afficher l'aperçu
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        imagePreview.classList.remove('d-none');
        uploadArea.classList.add('d-none');
    };
    reader.readAsDataURL(file);
}

// Analyser l'image
async function analyzeImage() {
    if (!currentImageFile) {
        showAlert('Veuillez d\'abord sélectionner une image', 'warning');
        return;
    }

    // Afficher le chargement
    imagePreview.classList.add('d-none');
    loadingSection.classList.remove('d-none');

    // Préparer les données du formulaire
    const formData = new FormData();
    formData.append('file', currentImageFile);

    try {
        // Envoyer l'image au serveur
        const response = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Afficher les résultats
            displayResults(data);
        } else {
            throw new Error(data.error || 'Erreur lors de l\'analyse');
        }

    } catch (error) {
        console.error('Erreur:', error);
        showAlert(`Erreur: ${error.message}`, 'danger');
        resetAnalysis();
    } finally {
        loadingSection.classList.add('d-none');
    }
}

// Afficher les résultats
function displayResults(data) {
    // Mettre à jour les informations générales
    analyzedImage.src = data.display_image;
    processingTime.textContent = data.processing_time;
    detectedCount.textContent = data.summary.present_count;

    // Effacer les anciens résultats
    attributesGrid.innerHTML = '';

    // Ajouter les cartes d'attributs
    data.results.forEach(result => {
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4';
        col.innerHTML = result.card_html;
        attributesGrid.appendChild(col);
    });

    // Animer l'apparition
    setTimeout(() => {
        const cards = document.querySelectorAll('.attribute-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';

            setTimeout(() => {
                card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }, 100);

    // Afficher la section des résultats
    resultsSection.classList.remove('d-none');
    resultsSection.classList.add('success-animation');
}

// Réinitialiser l'analyse
function resetAnalysis() {
    currentImageFile = null;
    fileInput.value = '';

    // Cacher les sections
    resultsSection.classList.add('d-none');
    imagePreview.classList.add('d-none');
    loadingSection.classList.add('d-none');

    // Réafficher la zone d'upload
    uploadArea.classList.remove('d-none');

    // Réinitialiser l'aperçu
    previewImage.src = '';

    // Réinitialiser les compteurs
    detectedCount.textContent = '0';
}

// Effacer l'image actuelle
function clearImage() {
    currentImageFile = null;
    fileInput.value = '';
    previewImage.src = '';
    imagePreview.classList.add('d-none');
    uploadArea.classList.remove('d-none');
}

// Télécharger le rapport
function downloadReport() {
    // Créer un rapport HTML simple
    const reportContent = `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'analyse - CelebA Detector</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .image-container { text-align: center; margin: 20px 0; }
                .attributes-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .attributes-table th, .attributes-table td {
                    border: 1px solid #ddd; padding: 12px; text-align: left;
                }
                .attributes-table th { background-color: #f2f2f2; }
                .present { color: green; }
                .absent { color: red; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Rapport d'analyse - CelebA Multi-Attribute Detector</h1>
                <p>Généré le ${new Date().toLocaleDateString()} à ${new Date().toLocaleTimeString()}</p>
            </div>

            <div class="image-container">
                <img src="${analyzedImage.src}" style="max-width: 300px;">
            </div>

            <div class="summary">
                <h3>Résumé</h3>
                <p><strong>Temps d'analyse:</strong> ${processingTime.textContent}</p>
                <p><strong>Attributs détectés:</strong> ${detectedCount.textContent} sur ${totalCount.textContent}</p>
            </div>

            <h3>Détails des attributs</h3>
            <table class="attributes-table">
                <thead>
                    <tr>
                        <th>Attribut</th>
                        <th>Présent</th>
                        <th>Confiance</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    ${Array.from(document.querySelectorAll('.attribute-card')).map(card => {
                        const attribute = card.querySelector('h5').textContent;
                        const status = card.querySelector('.attribute-status').textContent;
                        const confidence = card.querySelector('.confidence-text').textContent;
                        const description = card.querySelector('.attribute-description').textContent;
                        const isPresent = status.includes('Present');

                        return `
                            <tr>
                                <td><strong>${attribute}</strong></td>
                                <td class="${isPresent ? 'present' : 'absent'}">${isPresent ? '✓ Présent' : '✗ Absent'}</td>
                                <td>${confidence}</td>
                                <td>${description}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>

            <div style="margin-top: 30px; font-size: 0.9em; color: #666; text-align: center;">
                <p>Rapport généré par CelebA Multi-Attribute Detector</p>
                <p>Modèle: EfficientNet-B0 multi-tâches | Accuratie: 94.5%</p>
            </div>
        </body>
        </html>
    `;

    // Créer et télécharger le fichier
    const blob = new Blob([reportContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `celeba_analysis_${new Date().toISOString().slice(0,10)}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Afficher une alerte
function showAlert(message, type = 'info') {
    // Créer l'alerte
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = `
        top: 100px;
        right: 20px;
        z-index: 1050;
        min-width: 300px;
        max-width: 500px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;

    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Ajouter au document
    document.body.appendChild(alertDiv);

    // Supprimer automatiquement après 5 secondes
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// API functions
async function getModelInfo() {
    try {
        const response = await fetch(API_ENDPOINTS.modelInfo);
        return await response.json();
    } catch (error) {
        console.error('Erreur API:', error);
        return null;
    }
}

// Initialiser les informations du modèle
getModelInfo().then(info => {
    if (info) {
        console.log('Modèle chargé:', info.model_name);
    }
});