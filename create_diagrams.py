import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper function to set up a blank figure
def create_figure(title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    return fig, ax, filename

# Helper function to draw a box
def draw_box(ax, x, y, w, h, text, color='#E3F2FD'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Helper function to draw an arrow
def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', fontsize=9)

# --- 1. SYSTEM ARCHITECTURE ---
def draw_architecture():
    fig, ax, fname = create_figure("System Architecture", "system_architecture.png")
    
    # Nodes
    draw_box(ax, 0.5, 2.5, 2, 1, "User Input\n(Text / URL)")
    draw_box(ax, 3.5, 2.5, 2, 1, "Preprocessing\n(Cleaning & Tokenization)")
    draw_box(ax, 6.5, 2.5, 2, 1, "Feature Extraction\n(TF-IDF)")
    draw_box(ax, 3.5, 0.5, 2, 1, "ML Model\n(Passive Aggressive)")
    draw_box(ax, 6.5, 0.5, 2, 1, "Output / Dashboard\n(Real vs Fake)")

    # Arrows
    draw_arrow(ax, 2.5, 3.0, 3.5, 3.0) # Input -> Pre
    draw_arrow(ax, 5.5, 3.0, 6.5, 3.0) # Pre -> TFIDF
    
    # Arrow TFIDF -> Model (Down & Left)
    ax.annotate("", xy=(5.5, 1.0), xytext=(7.5, 2.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=0", lw=1.5))
    
    draw_arrow(ax, 3.5, 1.0, 2.5, 1.0, "") # Model -> (Self loop logic implied) ... let's connect model to output
    draw_arrow(ax, 5.5, 1.0, 6.5, 1.0) # Model -> Output

    plt.savefig(fname, bbox_inches='tight')
    print(f"Generated {fname}")
    plt.close()

# --- 2. DFD LEVEL 0 (CONTEXT) ---
def draw_dfd0():
    fig, ax, fname = create_figure("DFD Level 0: Context Diagram", "dfd_level_0.png")
    
    # External Entity (User)
    draw_box(ax, 0.5, 2.5, 2, 1, "User", color='#FFEBEE')
    
    # Process (System)
    circle = patches.Circle((7, 3), radius=1.2, edgecolor='black', facecolor='#E8F5E9', lw=1.5)
    ax.add_patch(circle)
    ax.text(7, 3, "0.0\nFake News\nDetection System", ha='center', va='center', fontweight='bold')
    
    # Arrows
    # User -> System
    ax.annotate("", xy=(5.8, 3.2), xytext=(2.5, 3.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(4.15, 3.3, "News Text / URL", ha='center', fontsize=9)
    
    # System -> User
    ax.annotate("", xy=(2.5, 2.8), xytext=(5.8, 2.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(4.15, 2.5, "Prediction Result", ha='center', fontsize=9)
    
    plt.savefig(fname, bbox_inches='tight')
    print(f"Generated {fname}")
    plt.close()

# --- 3. DFD LEVEL 1 ---
def draw_dfd1():
    fig, ax, fname = create_figure("DFD Level 1: Detailed Flow", "dfd_level_1.png")
    
    draw_box(ax, 0.5, 4.5, 1.8, 0.8, "1.0\nInput Handler")
    draw_box(ax, 3.0, 4.5, 1.8, 0.8, "2.0\nText Cleaner")
    draw_box(ax, 5.5, 4.5, 1.8, 0.8, "3.0\nVectorizer")
    draw_box(ax, 8.0, 4.5, 1.5, 0.8, "4.0\nModel")
    draw_box(ax, 4.0, 2.0, 2.0, 0.8, "5.0\nResult Display")
    
    draw_arrow(ax, 2.3, 4.9, 3.0, 4.9)
    draw_arrow(ax, 4.8, 4.9, 5.5, 4.9)
    draw_arrow(ax, 7.3, 4.9, 8.0, 4.9)
    
    # Model -> Display
    ax.annotate("", xy=(6.0, 2.4), xytext=(8.75, 4.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=0", lw=1.5))
    
    plt.savefig(fname, bbox_inches='tight')
    print(f"Generated {fname}")
    plt.close()

# --- 4. USE CASE DIAGRAM ---
def draw_usecase():
    fig, ax, fname = create_figure("Use Case Diagram", "use_case_diagram.png")
    
    # Stick Figure
    ax.text(1.0, 3.0, "웃", fontsize=50, ha='center')
    ax.text(1.0, 1.8, "User", ha='center', fontweight='bold')
    
    # System Boundary
    rect = patches.Rectangle((3, 0.5), 6, 5, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(6, 5.2, "Fake News Detection App", ha='center', fontweight='bold')
    
    # Use Cases
    cases = [(6, 4.2, "Input Text/URL"), (6, 3.2, "View Prediction"), (6, 2.2, "View Analytics"), (6, 1.2, "Retrain Model")]
    
    for x, y, text in cases:
        ellipse = patches.Ellipse((x, y), 3.5, 0.8, edgecolor='black', facecolor='white', lw=1.5)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha='center', va='center')
        # Line from User
        ax.plot([1.5, x-1.75], [3.2, y], color='black', lw=1)

    plt.savefig(fname, bbox_inches='tight')
    print(f"Generated {fname}")
    plt.close()

# --- 5. CLASS DIAGRAM ---
def draw_class():
    fig, ax, fname = create_figure("Class Diagram", "class_diagram.png")
    
    # Class Boxes (UML style)
    def draw_uml_class(x, y, name, attribs, methods):
        # Header
        draw_box(ax, x, y+1.5, 2.5, 0.5, name, color='#FFE082')
        # Body
        rect = patches.Rectangle((x, y), 2.5, 1.5, edgecolor='black', facecolor='white', lw=1.5)
        ax.add_patch(rect)
        ax.text(x+0.1, y+1.3, attribs, va='top', fontsize=9)
        # Separator line
        ax.plot([x, x+2.5], [y+0.75, y+0.75], color='black', lw=1)
        ax.text(x+0.1, y+0.6, methods, va='top', fontsize=9)

    draw_uml_class(1, 3, "StreamlitApp", "- input_text: String\n- model: Object", "+ render()\n+ get_input()\n+ show_result()")
    draw_uml_class(4.5, 3, "ModelEngine", "- vectorizer: Tfidf\n- classifier: PAC", "+ load_data()\n+ train()\n+ predict(text)")
    draw_uml_class(4.5, 0.5, "Preprocessing", "- stopwords: List\n- regex: String", "+ clean_text()\n+ tokenize()")
    
    # Relationships
    draw_arrow(ax, 3.5, 4.0, 4.5, 4.0, "uses")
    draw_arrow(ax, 5.75, 3.0, 5.75, 2.0, "uses")

    plt.savefig(fname, bbox_inches='tight')
    print(f"Generated {fname}")
    plt.close()

if __name__ == "__main__":
    draw_architecture()
    draw_dfd0()
    draw_dfd1()
    draw_usecase()
    draw_class()
    print("\n✅ All 5 diagrams generated successfully!")