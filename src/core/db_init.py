# src/core/db_init.py
from sqlalchemy import create_engine, text
from .config_manager import config_manager
import logging

logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize database with SDG schema and seed data"""
    engine = create_engine(config_manager.get_database_url())

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.sdgs (
              goal_number   integer PRIMARY KEY,
              name          text    NOT NULL,
              color_hex     text    NOT NULL,
              description   text    NOT NULL
            );
        """))

        sdg_data = [
            (1,  "No Poverty",                          "#E5243B", "End poverty in all its forms everywhere"),
            (2,  "Zero Hunger",                         "#DDA63A", "End hunger, achieve food security and improved nutrition and promote sustainable agriculture"),
            (3,  "Good Health and Well-Being",          "#4C9F38", "Ensure healthy lives and promote well-being for all at all ages"),
            (4,  "Quality Education",                   "#C5192D", "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all"),
            (5,  "Gender Equality",                     "#FF3A21", "Achieve gender equality and empower all women and girls"),
            (6,  "Clean Water and Sanitation",          "#26BDE2", "Ensure availability and sustainable management of water and sanitation for all"),
            (7,  "Affordable and Clean Energy",         "#FBC412", "Ensure access to affordable, reliable, sustainable and modern energy for all"),
            (8,  "Decent Work and Economic Growth",     "#A21942", "Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all"),
            (9,  "Industry, Innovation and Infrastructure", "#FD6925", "Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation"),
            (10, "Reduced Inequalities",                "#DD1367", "Reduce inequality within and among countries"),
            (11, "Sustainable Cities and Communities",  "#F89D2A", "Make cities and human settlements inclusive, safe, resilient and sustainable"),
            (12, "Responsible Consumption and Production", "#BF8D2C", "Ensure sustainable consumption and production patterns"),
            (13, "Climate Action",                      "#3F7E44", "Take urgent action to combat climate change and its impacts"),
            (14, "Life Below Water",                    "#0A97D9", "Conserve and sustainably use the oceans, seas and marine resources for sustainable development"),
            (15, "Life on Land",                        "#56C02B", "Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss"),
            (16, "Peace, Justice and Strong Institutions", "#00689D", "Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels"),
            (17, "Partnerships for the Goals",          "#19486A", "Strengthen the means of implementation and revitalize the Global Partnership for Sustainable Development"),
        ]


        for goal_num, name, color, desc in sdg_data:
            conn.execute(text("""
                INSERT INTO sdgs (goal_number, name, color_hex, description)
                VALUES (:goal_num, :name, :color, :desc)
                ON CONFLICT (goal_number) DO NOTHING
            """), {"goal_num": goal_num, "name": name, "color": color, "desc": desc})

    logger.info("Database initialized with SDG schema and data")

if __name__ == "__main__":
    initialize_database()
