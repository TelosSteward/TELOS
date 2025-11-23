#!/usr/bin/env python3
"""
Create TELOS Extension Icons
Cyberpunk-themed gradient icons matching the extension UI
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a TELOS-branded icon with Observatory color scheme"""

    # Create image with dark background
    img = Image.new('RGBA', (size, size), (26, 26, 26, 255))  # #1a1a1a
    draw = ImageDraw.Draw(img)

    # Draw elevated circle in center (#2d2d2d)
    margin = size // 8
    circle_bbox = [margin, margin, size - margin, size - margin]
    draw.ellipse(circle_bbox, fill=(45, 45, 45, 255))

    # Draw "T" letter in center (scaled to icon size)
    if size >= 48:
        # For larger icons, draw a proper "T"
        font_size = size // 2
        try:
            # Try to use a bold system font
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()

        text = "T"
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        x = (size - text_width) // 2 - bbox[0]
        y = (size - text_height) // 2 - bbox[1]

        # Draw text with TELOS gold (#F4D03F = RGB 244, 208, 63)
        draw.text((x, y), text, fill=(244, 208, 63, 255), font=font)
    else:
        # For 16px, draw a simple shape
        # Draw a "T" shape using rectangles
        bar_width = size // 3
        bar_height = size // 8
        vertical_height = size // 2

        # Horizontal bar of T
        top_y = size // 3
        draw.rectangle(
            [margin * 2, top_y, size - margin * 2, top_y + bar_height],
            fill=(244, 208, 63, 255)
        )

        # Vertical bar of T
        center_x = size // 2
        draw.rectangle(
            [center_x - bar_height // 2, top_y, center_x + bar_height // 2, top_y + vertical_height],
            fill=(244, 208, 63, 255)
        )

    # Add subtle gold glow effect
    for i in range(2):
        alpha = 40 - i * 15
        offset = i + 1
        draw.ellipse(
            [margin - offset, margin - offset, size - margin + offset, size - margin + offset],
            outline=(244, 208, 63, alpha),
            width=1
        )

    # Save
    img.save(output_path, 'PNG')
    print(f"✓ Created {output_path} ({size}x{size})")

def main():
    # Get the extension icons directory
    icons_dir = "/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOS_Extension/icons"

    # Create icons in all required sizes
    sizes = [
        (16, 'icon16.png'),
        (48, 'icon48.png'),
        (128, 'icon128.png')
    ]

    for size, filename in sizes:
        output_path = os.path.join(icons_dir, filename)
        create_icon(size, output_path)

    print("\n✅ All icons created successfully!")
    print(f"📁 Location: {icons_dir}")
    print("\n🎨 Icons match TELOS Observatory color scheme:")
    print("   - Dark background (#1a1a1a)")
    print("   - Muted gold 'T' branding (#F4D03F)")
    print("   - Subtle gold glow effect")
    print("   - Professional, accessible design")

if __name__ == '__main__':
    main()
