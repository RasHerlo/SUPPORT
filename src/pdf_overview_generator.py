import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image
import argparse
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT

def get_image_size(image_path, max_width=3*inch, max_height=3*inch):
    """Get scaled image dimensions while maintaining aspect ratio."""
    with Image.open(image_path) as img:
        width, height = img.size
        aspect_ratio = width / height
        
        if width > max_width:
            width = max_width
            height = width / aspect_ratio
            
        if height > max_height:
            height = max_height
            width = height * aspect_ratio
            
        return width, height

def wrap_text(canvas, text, x, y, max_width):
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        word_width = canvas.stringWidth(' '.join(current_line + [word]), "Helvetica-Bold", 14)
        if word_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def generate_pdf_overview(parent_folder, output_filename="pdf_overview_SUPPORT_files.pdf"):
    """Generate a PDF overview of SUPPORT processed images."""
    print(f"Starting PDF generation from folder: {parent_folder}")
    
    # Create PDF canvas
    output_path = os.path.join(parent_folder, output_filename)
    print(f"PDF will be saved as: {output_path}")
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    
    # Set margins and initial position
    left_margin = 1*inch
    right_margin = 1*inch
    top_margin = height - 1*inch
    bottom_margin = 1*inch
    current_y = top_margin
    line_height = 0.3*inch
    image_spacing = 0.2*inch
    
    # Calculate available width for content
    content_width = width - left_margin - right_margin
    
    # Walk through the parent folder
    for root, dirs, files in os.walk(parent_folder):
        print(f"Checking directory: {root}")
        if os.path.basename(root) == "DATA":
            print(f"Found DATA folder: {root}")
            
            # Add main heading with wrapping
            current_y -= line_height
            c.setFont("Helvetica-Bold", 14)
            heading_text = f"Data Folder: {os.path.dirname(root)}"
            lines = wrap_text(c, heading_text, left_margin, current_y, content_width)
            for line in lines:
                c.drawString(left_margin, current_y, line)
                current_y -= line_height
            
            # Check for SUPPORT folders
            support_folders = [d for d in dirs if d.startswith("SUPPORT_")]
            print(f"Found SUPPORT folders: {support_folders}")
            
            if support_folders:
                # Calculate image sizes for side-by-side placement
                max_image_width = (content_width - image_spacing) / 2
                max_image_height = 3*inch
                
                # Get all images first
                images = []
                for support_folder in support_folders:
                    image_path = os.path.join(root, support_folder, "denoised_cut_avr.png")
                    print(f"Looking for image: {image_path}")
                    if os.path.exists(image_path):
                        print(f"Found image: {image_path}")
                        img_width, img_height = get_image_size(image_path, max_image_width, max_image_height)
                        images.append((support_folder, image_path, img_width, img_height))
                
                # Place images side by side
                if images:
                    # Add subheadings
                    current_y -= line_height
                    c.setFont("Helvetica", 12)
                    for i, (folder, _, _, _) in enumerate(images):
                        x = left_margin + i * (max_image_width + image_spacing)
                        c.drawString(x, current_y, folder)
                    
                    current_y -= line_height
                    
                    # Check if we need a new page
                    if current_y - max_image_height - image_spacing < bottom_margin:
                        c.showPage()
                        current_y = top_margin
                    
                    # Draw images
                    for i, (_, image_path, img_width, img_height) in enumerate(images):
                        x = left_margin + i * (max_image_width + image_spacing)
                        c.drawImage(image_path, x, current_y - img_height, 
                                  width=img_width, height=img_height)
                    
                    current_y -= max_image_height + image_spacing
            
            # Add spacing between data folders
            current_y -= line_height
            
            # Check if we need a new page
            if current_y < bottom_margin:
                c.showPage()
                current_y = top_margin
    
    # Save the PDF
    c.save()
    print(f"PDF overview generated: {os.path.join(parent_folder, output_filename)}")

def main():
    parser = argparse.ArgumentParser(description='Generate PDF overview of SUPPORT processed images')
    parser.add_argument('--parent_folder', type=str, required=True,
                       help='Parent folder containing DATA folders with SUPPORT processed images')
    parser.add_argument('--output', type=str, default="pdf_overview_SUPPORT_files.pdf",
                       help='Output PDF filename')
    
    args = parser.parse_args()
    print(f"Parent folder: {args.parent_folder}")
    print(f"Output filename: {args.output}")
    generate_pdf_overview(args.parent_folder, args.output)

if __name__ == '__main__':
    main() 