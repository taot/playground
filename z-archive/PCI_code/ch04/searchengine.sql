drop database if exists searchengine;
create database searchengine;

use searchengine;

create table urllist (
    rowid               bigint not null auto_increment,
    url                 varchar(500),
    primary key (rowid),
    index (url)
) default character set = utf8;

create table wordlist (
    rowid               bigint not null auto_increment,
    word                varchar(50),
    primary key (rowid),
    index (word)
) default character set = utf8;

create table wordlocation (
    rowid               bigint not null auto_increment,
    urlid               bigint not null,
    wordid              bigint not null,
    location            bigint not null,
    primary key (rowid),
    constraint foreign key (urlid) references urllist (rowid),
    constraint foreign key (wordid) references wordlist (rowid)
) default character set = utf8;

create table link (
    rowid               bigint not null auto_increment,
    fromid              bigint not null,
    toid                bigint not null,
    primary key (rowid),
    constraint foreign key (fromid) references urllist (rowid),
    constraint foreign key (toid) references urllist (rowid)
) default character set = utf8;

create table linkwords (
    rowid               bigint not null auto_increment,
    wordid              bigint not null,
    linkid              bigint not null,
    primary key (rowid),
    constraint foreign key (wordid) references wordlist (rowid),
    constraint foreign key (linkid) references link (rowid)
) default character set = utf8;
