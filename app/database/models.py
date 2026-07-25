from sqlalchemy import Column, Integer
from . import get_base

Base = get_base()


class Images(Base):
    __tablename__ = "inferences"

    id = Column(Integer, primary_key=True, index=True)
