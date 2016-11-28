drop database if exists nn;
create database nn;

use nn;

create table hiddennode(
    rowid               bigint not null auto_increment,
    create_key          varchar(200) not null,
    primary key (rowid)
) default character set = utf8;

create table wordhidden(
    rowid               bigint not null auto_increment,
    fromid              bigint not null,
    toid                bigint not null,
    strength            decimal(20,10) not null,
    primary key (rowid)
) default character set = utf8;

create table hiddenurl(
    rowid               bigint not null auto_increment,
    fromid              bigint not null,
    toid                bigint not null,
    strength            decimal(20,10) not null,
    primary key (rowid)
) default character set = utf8;
