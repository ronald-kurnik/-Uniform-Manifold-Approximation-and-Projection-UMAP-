import umap
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import load_digits
from PIL import Image
import io
import base64

# -------------------------------
# 1. Load data + UMAP
# -------------------------------
digits = load_digits()
X = digits.data
y = digits.target

reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, random_state=42)
embedding = reducer.fit_transform(X)

# -------------------------------
# 2. Create hover images (base64)
# -------------------------------
hover_images = []
for i in range(len(X)):
    img = X[i].reshape(8, 8)
    img = (img / 16.0 * 255).astype('uint8')
    img_pil = Image.fromarray(img).resize((80, 80), Image.NEAREST)
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    hover_images.append(img_str)

# -------------------------------
# 3. Build SINGLE trace (critical!)
# -------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=embedding[:, 0],
    y=embedding[:, 1],
    mode='markers',
    marker=dict(
        size=9,
        color=y,
        colorscale='Spectral',
        line=dict(width=0.5, color='white')
    ),
    text=[f"<b>Label: {label}</b><br>Index: {i}" for i, label in enumerate(y)],
    hoverinfo='text',
    showlegend=False
))

# -------------------------------
# 4. Add images via `layout.images` (this is the trick!)
# -------------------------------
images = []
for i in range(len(X)):
    x, y_pos = embedding[i]
    images.append(dict(
        source=f"data:image/png;base64,{hover_images[i]}",
        xref="x",
        yref="y",
        x=x,
        y=y_pos,
        sizex=2.5,   # Adjust size
        sizey=2.5,
        xanchor="center",
        yanchor="middle",
        opacity=0,
        layer="above"
    ))

fig.update_layout(
    images=images,
    title="UMAP Digits — Hover to Reveal Handwritten Digit!",
    title_font_size=20,
    width=1000,
    height=800,
    hovermode="closest",
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    template="simple_white"
)

# -------------------------------
# 5. Custom hover: show image on hover
# -------------------------------
fig.update_traces(
    hovertemplate=(
        "%{text}<br>" +
        "<img src='data:image/png;base64,%{customdata}' width='80' height='80'>"
    ),
    customdata=hover_images
)

# -------------------------------
# 6. JavaScript to show image only on hover
# -------------------------------
fig_html = fig.to_html(include_plotlyjs="cdn")

# Inject JS to show image only on hover
js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('image');
    images.forEach(img => img.style.opacity = '0');
    
    Plotly.d3.select('.plotly').on('plotly_hover', function(d) {
        const point = d.points[0];
        const idx = point.pointIndex;
        images[idx].style.opacity = '1';
    });
    
    Plotly.d3.select('.plotly').on('plotly_unhover', function() {
        images.forEach(img => img.style.opacity = '0');
    });
});
</script>
"""

final_html = fig_html.replace("</body>", js + "</body>")

# -------------------------------
# 7. Save as HTML
# -------------------------------
with open("umap_digits_hover.html", "w") as f:
    f.write(final_html)

print("SUCCESS! Open 'umap_digits_hover.html' in your browser.")
print("Hover over any point → the digit appears!")