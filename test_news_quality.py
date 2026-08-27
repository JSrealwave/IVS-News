#!/usr/bin/env python3
from datetime import datetime, timezone

from news_quality import (
    canonicalize_article_url,
    is_cfp_or_tagline,
    is_generic_seo_mill,
    is_heading_title,
    is_product_landing,
    is_quality_article_image,
    reject_reason,
    tavily_time_range,
)


def test_tavily_range() -> None:
    monday = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)  # Monday
    tuesday = datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc)
    assert tavily_time_range(monday) == "week"
    assert tavily_time_range(tuesday) == "d"
    friday = datetime(2026, 8, 28, 13, 0, tzinfo=timezone.utc)
    assert tavily_time_range(friday) == "d"


def test_canonical() -> None:
    assert (
        canonicalize_article_url(
            "http://www.forasoft.com/blog/article/x?utm_source=x&keep=1"
        )
        == "https://forasoft.com/blog/article/x?keep=1"
    )
    assert (
        canonicalize_article_url("https://arxiv.org/html/2607.24904v1")
        == "https://arxiv.org/abs/2607.24904"
    )


def test_rejects() -> None:
    assert is_heading_title("Introduction")
    assert is_heading_title("Introduction — VSS")
    assert is_heading_title("Introduction | The 2026 Edge AI Technology Report - Wevolver")
    assert is_cfp_or_tagline(
        "The premier conference for innovators incorporating computer vision and physical AI in products",
        "https://embeddedvisionsummit.com/call-proposals",
    )
    assert is_product_landing(
        "Security — Perimeter & Intrusion Detection with 3D LiDAR — Metrolla®",
        "https://metrolla.com/solutions/security",
    )
    assert is_product_landing(
        "3D LiDAR Perimeter Security: PulseVi FAQs | RBtec",
        "https://www.rbtec.com/pulsevi-faqs",
    )
    assert is_product_landing(
        "AI Mining Safety Monitoring with Computer Vision | Appther",
        "https://www.appther.com/blogs/ai-mining-safety-monitoring-computer-vision",
    )
    assert is_generic_seo_mill(
        "AI Security Cameras in 2026: IP Camera Buyer's Guide",
        "https://example.com/guide",
    )
    assert not is_generic_seo_mill(
        "Real-Time Video Processing with AI: The 2026 Playbook",
        "https://www.forasoft.com/blog/article/real-time-video-processing-with-ai",
    )
    assert reject_reason("Introduction", "https://arxiv.org/html/2607.24904v1")
    assert reject_reason(
        "Introduction — VSS", "https://docs.nvidia.com/vss/latest"
    )


def test_images() -> None:
    assert not is_quality_article_image(
        "https://arxiv.org/static/browse/0.3.4/images/icons/apple-touch-icon.png"
    )
    assert not is_quality_article_image(
        "https://www.rbtec.com/wp-content/uploads/2020/02/cropped-favicon-180x180.jpg"
    )
    assert not is_quality_article_image("https://example.com/logo/mark.png")
    assert is_quality_article_image(
        "https://benchmarkmagazine.com/wp-content/uploads/2026/07/Beyond-megapixels-4.png"
    )


if __name__ == "__main__":
    test_tavily_range()
    test_canonical()
    test_rejects()
    test_images()
    print("test_news_quality: ok")
